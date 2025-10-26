import ast
import json
import os
import shutil
import time
from copy import deepcopy
from typing import Dict, Tuple

import joblib
import typer
import yaml

from dbinfer_bench.dataset_meta import DBBColumnDType, DBBColumnSchema
from dbinfer_bench.rdb_dataset import DBBRDBDataset

from ...prompts.autom import get_autom_action_selection_prompt
from ...prompts.mautog import (
    get_multi_round_action_selection_prompt,
    get_single_round_multi_step_prompt,
)
from ...utils import get_logger
from ...utils.misc import copy_directory
from ...utils.plot import plot_rdb_dataset_schema
from ..autog.action import (
    get_autog_actions,
    pack_function_introduction_prompt,
    turn_dbb_into_a_lookup_table,
)
from ..autog.deepjoin import join_discovery, load_pretrain_jtd_lm
from ..llm.gconstruct import (
    analyze_dataframes,
    dummy_llm_interaction,
    extract_between_tags,
)

logger = get_logger(__name__)
import traceback
from pathlib import Path


def ordinal(n: int) -> str:
    """Returns the ordinal form of a number (e.g., 1st, 2nd, 3rd, 4th, etc.)."""
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    suffix = suffixes.get(n % 10, "th") if n % 10 in suffixes or n % 100 // 10 == 1 else "th"
    return str(n) + suffix


def load_dbb_dataset_from_cfg_path_no_name(cfg_path: str):
    return DBBRDBDataset(Path(cfg_path))


class AutoM_Agent:
    def __init__(
        self,
        initial_schema,
        mode="autog-s",
        oracle=None,
        path_to_file="",
        use_cache=False,
        threshold=10,
        llm_sleep=0.5,
        task_description="autog",
        dataset="mag",
        task_name="venue",
        schema_info="",
        lm_path="",
        jtd_k=20,
        recalculate=True,
        data_type_file="",
        update_task=False,
    ) -> None:
        """
        Main agent program for AutoG
        Args:
            initial_schema: dict: the initial schema of the data inferred by llms
            mode: str: the mode of the agent, either autog-a or autog-s. autog-a will generate several candidates while autog-s will use the last state as the output
            oracle: the oracle model to evaluate the schema
            llm_model_name: str: the name of the llm model
            context_size: int: the context size of the llm model
            path_to_file: str: the path to the file
            llm_sleep: int: the sleep time of the llm api call
            use_cache: bool: whether to use cache for llm calling
            threshold: int: the maximum number of running rounds for autog
            task_description: str: the task description of the current task
            dataset: str: the dataset name
            task_name: str: the task name
            schema_info: str: the schema information
            lm_path: str: the path to the pre-trained deep join model
            jtd_k: int: the number of top k similar columns
            recalculate: bool: whether to recalculate the deep join and statistics for each round
            data_type_file: str: the file to store the data type
            update_task: bool: whether to update the task after each round, for relbench, no need
        """
        self.llm = ""
        self.action_list = get_autog_actions()
        self.threshold = threshold
        self.state = initial_schema
        self.original_state = deepcopy(initial_schema)
        self.mode = mode
        self.oracle = oracle
        self.icl_strategy = "random"
        self.path_to_file = path_to_file
        self.use_cache = use_cache
        self.dataset = dataset
        ## eval mode: whether eval each step or eval at the end ('step', 'end')
        self.error = 0
        self.success = 0
        ## find the file name of the target table
        ## storing an object:
        ## index
        ## path to saved objects
        self.task_name = task_name
        self.task_description = task_description
        self.schema_info = schema_info
        self.lm_path = lm_path
        self.jtd_k = jtd_k
        self.recalculate = recalculate
        self.round = 0
        self.autom_path = None

        ## by default, we use the default prompts
        self.icl_demonstrations = []
        self.history = []
        self.llm_sleep = llm_sleep
        self.data_type_file = data_type_file
        self.need_update_task = update_task
        # if self.mode == 'autog-s':
        examples = get_single_round_multi_step_prompt()
        for example in examples:
            self.icl_demonstrations.append(example)

    def backup(self, dbb):
        """
        Backup the current state
        """
        dbb.save(os.path.join(self.path_to_file, f"backup_{len(self.history)}"))

    def retrieve_icl_samples(self):
        """
        Retrieve the ICL samples
        """
        return self.icl_demonstrations

    def pack_prompts(self, dbb):
        """
        Generate the AutoG prompts
        args:
            update: bool: whether to update the stats after changing the schema
        """
        history_str = (
            "\n\n".join(self.history)
            if len(self.history) > 0
            else "First iteration, no history yet\n\n"
        )
        ## get the schema
        schema = dbb.metadata.json()
        if not self.recalculate:
            stats = self.schema_info
        else:
            table_meta_dict = {
                f"Table {table_name}": table for table_name, table in dbb.tables.items()
            }
            if self.dataset == "stackexchange":
                stats = analyze_dataframes(table_meta_dict, dbb=dbb)
            else:
                stats = analyze_dataframes(table_meta_dict)
        # full_prompts = get_multi_round_action_selection_prompt(
        #     action_description,
        #     example_str,
        #     history_str,
        #     schema,
        #     stats,
        #     self.task_description,
        #     '',
        # )

        ## save the current prompt for debug
        return get_autom_action_selection_prompt(
            history_actions=history_str,
            input_schema=schema,
            stats=stats,
            task=self.task_description,
        )

    def parse_args(self, parameters):
        return ast.literal_eval(parameters)

    def decide_next_step(self, dbb: DBBRDBDataset, llm_path):
        """
        Input the current state, let LLMs determine the next action
        """
        # import ipdb; ipdb.set_trace()
        selection = "nothing selected yet"
        # import ipdb; ipdb.set_trace()
        this_round_dbb = dbb
        this_round_prompt = self.pack_prompts(this_round_dbb)
        # response = bedrock_llm_query(self.llm, this_round_prompt, max_tokens = self.output_size, cache=self.use_cache, debug_dataset=self.dataset, debug_task=self.task_name, debug_round=epoch-1)
        query_file_path = os.path.join(llm_path, "query.txt")
        response_file_path = os.path.join(llm_path, "response.txt")
        response = dummy_llm_interaction(this_round_prompt, query_file_path, response_file_path)
        logger.info(response_file_path)
        selection = extract_between_tags(response, "selection")[0].strip()
        if selection == "None":
            return this_round_dbb, False
        # method = extract_between_tags(response, "construction")[0].strip()
        ## update here, no need to let llm tell method, just try r2n and r2ne both
        method = "r2n"

        ## use a for loop to run these commands
        selection = json.loads(selection)
        # import ipdb; ipdb.set_trace()

        # register duckdb
        import glob

        import duckdb
        import numpy as np
        import pandas as pd

        from dbinfer_bench.dataset_meta import DBBTableDataFormat, DBBTableSchema
        from dbinfer_bench.table_loader import get_table_data_loader

        parquet_files = glob.glob(os.path.join(self.path_to_file, "data", "*.pqt"))
        logger.info(f"{parquet_files}")
        for f in parquet_files:
            # table name = filename without extension
            table_name = os.path.splitext(os.path.basename(f))[0]
            logger.info(table_name)
            locals()[table_name] = pd.read_parquet(f)
            # duckdb.query(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{f}')")
        loader = get_table_data_loader(DBBTableDataFormat.PARQUET)

        for move in selection:
            logger.info(f"Move: {move}")
            last_valid_dbb = deepcopy(this_round_dbb)
            explanation = move["explanation"]
            methods = move["action"]
            parameters = move["parameters"]
            sqls = move["sqls"]
            new_table_schema = move["new_table"]
            logger.info(f"new_table_schema: {new_table_schema}")

            parameters["dbb"] = this_round_dbb
            try:
                new_table_name = new_table_schema["name"]
                Path(f"{self.autom_path}/data").mkdir(exist_ok=True, parents=True)
                new_tab_path = f"{self.autom_path}/data/{new_table_name}.pqt"
                t = duckdb.query(sqls[0])
                t.to_parquet(new_tab_path)
                t = pd.read_parquet(new_tab_path)
                np.savez_compressed(new_tab_path.replace(".pqt", ".npz"), **t)
                new_table = loader(new_tab_path)
                logger.info(f"new_table: {new_table}")
                this_round_dbb.tables[new_table_name] = new_table
                this_round_dbb.metadata.tables.append(
                    DBBTableSchema.parse_obj(
                        {"format": DBBTableDataFormat.NUMPY, **new_table_schema}
                    )
                )

                # this_round_dbb = action_code(**parameters)
                ## remove non-serializable objects
                move["parameters"] = {k: v for k, v in move["parameters"].items() if k != "dbb"}
                self.history.append(json.dumps(move))
                # this_round_dbb.method = method
                self.success += 1
                logger.info(f"DONE.")
            except Exception as e:
                ## recover from error
                logger.info(f"Error: {e}")
                traceback.print_exc()
                self.history.append("Error: " + str(e) + "Problem action: " + str(move))
                this_round_dbb = last_valid_dbb
                self.error += 1
                if self.error >= 3:
                    return this_round_dbb, False
        ## autog for relbench, quit after one round
        logger.info(f"final metadata: {this_round_dbb.metadata}")
        return this_round_dbb, False

    def get_current_state(self):
        """
        Get the current state
        """
        return self.state

    def manual_post_process(self, dbb):
        ## this is a manual post process function
        ## some datasets require time_column otherwise there will be leakage
        ## this has nothing with todo graph construction, only for evaluation purpose
        ## we put process here
        if self.dataset == "outbrain":
            for i, table in enumerate(dbb.metadata.tables):
                if table.name == "Event":
                    table.time_column = "timestamp"
                if table.name == "PageView":
                    table.time_column = "timestamp"
                if table.name == "Click":
                    table.time_column = "timestamp"
                if table.name == "DocumentsMeta":
                    table.time_column = "publish_time"
            dbb.metadata.tasks[0].time_column = "timestamp"
        if self.dataset == "avs":
            # dbb.tasks[0].train_set.pop('history_chain')
            # dbb.tasks[0].validation_set.pop('history_chain')
            # dbb.tasks[0].test_set.pop('history_chain')
            dbb.tasks[0].metadata.columns.append(
                DBBColumnSchema(
                    name="timestamp",
                    dtype=DBBColumnDType.datetime_t,
                )
            )
            dbb.tasks[0].time_column = "timestamp"
            remove_table = -1
            remove_col = -1
            for i, table in enumerate(dbb.metadata.tables):
                if table.name == "History":
                    table.time_column = "offerdate"
                if table.name == "Transaction":
                    table.time_column = "date"
                if table.name == "History":
                    for j, col in enumerate(table.columns):
                        if col.name == "repeater":
                            remove_table = i
                            remove_col = j
            if remove_table != -1 and remove_col != -1:
                dbb.metadata.tables[remove_table].columns.pop(remove_col)
            dbb.metadata.tasks[0].time_column = "timestamp"
            ## for history table, need to pop the repeater column, otherwise it will be a leakage; why for mag we don't need to remove? didn't figure out yet

        if self.dataset == "diginetica":
            #
            for i, table in enumerate(dbb.metadata.tables):
                if table.name == "QueryResult":
                    table.time_column = "timestamp"
                elif table.name == "Click":
                    table.time_column = "timestamp"
                elif table.name == "View":
                    table.time_column = "timestamp"
                elif table.name == "Purchase":
                    table.time_column = "timestamp"
                elif table.name == "Query":
                    table.time_column = "timestamp"
            if self.task_name == "purchase":
                dbb.metadata.tasks[0].time_column = "timestamp"
                dbb.metadata.tasks[0].columns = [
                    DBBColumnSchema(
                        name="timestamp",
                        dtype=DBBColumnDType.datetime_t,
                    ),
                    DBBColumnSchema(
                        name="purchase_session",
                        link_to="Session.sessionId",
                        dtype=DBBColumnDType.foreign_key,
                    ),
                    DBBColumnSchema(
                        name="itemId",
                        link_to="Product.itemId",
                        dtype=DBBColumnDType.foreign_key,
                    ),
                ]
            else:
                dbb.metadata.tasks[0].time_column = "timestamp"
                dbb.metadata.tasks[0].columns = [
                    DBBColumnSchema(
                        name="timestamp",
                        dtype=DBBColumnDType.datetime_t,
                    ),
                    DBBColumnSchema(
                        name="queryId",
                        link_to="Query.queryId",
                        dtype=DBBColumnDType.foreign_key,
                    ),
                    DBBColumnSchema(
                        name="itemId",
                        link_to="Product.itemId",
                        dtype=DBBColumnDType.foreign_key,
                    ),
                    DBBColumnSchema(name="clicked", dtype=DBBColumnDType.category_t),
                ]
        return dbb

    def fix_pk_fk(self, dbb):
        """
        Since in connect_two_columns, we make it possible to turn a pk in a fk. However, link_to can only be made towards a pk. As a result, we need to check every link_to to make sure that they are point to a pk.
        """

        for table in dbb.metadata.tables:
            for column in table.columns:
                pass

    def augment(self):
        """
        Augment the schema
        """
        logger.info("Start to augment the schema")
        dbb_root_path = self.path_to_file
        dbb = load_dbb_dataset_from_cfg_path_no_name(dbb_root_path)

        base_path = Path(self.path_to_file).parent.parent.parent.joinpath("autom", self.task_name)
        save_path = base_path / "final"
        save_path.mkdir(exist_ok=True, parents=True)
        self.autom_path = save_path

        plot_rdb_dataset_schema(dbb, save_path.joinpath("original-schema").__str__())
        for i in range(self.threshold):
            logger.info(f"Round: {i}")
            ## generate the folder for round i

            llm_path = base_path.joinpath(f"{self.task_name}_round_{i}")
            llm_path.mkdir(exist_ok=True, parents=True)
            res, need_continue = self.decide_next_step(dbb, llm_path)
            self.round += 1
            if need_continue == False:
                # import ipdb; ipdb.set_trace()
                res = self.manual_post_process(res)
                if self.error < 3:
                    logger.info("No more action can be taken")
                    # import ipdb; ipdb.set_trace()
                    res.save(save_path)
                    ## plot the schema
                    plot_rdb_dataset_schema(res, save_path.joinpath("schema").__str__())
                    return
                elif self.error >= 3:
                    logger.info("Too many errors, halt the process")
                    res.save(save_path)
                    plot_rdb_dataset_schema(res, save_path.joinpath("schema").__str__())
                    return
            else:
                dbb = res
                time.sleep(self.llm_sleep)
