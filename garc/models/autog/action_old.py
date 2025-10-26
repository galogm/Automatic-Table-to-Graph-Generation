"""
Update in the camera ready version:
Instead of loading the file from the disk, we directly pass dbb as a parameter and change everything
there, it will be much faster since the number of disk read/write will be reduced.
After this change, we don't need to hard_code function as string but can directly call them.
"""

import inspect


def apply_simulated_actions(function_name, params):
    actions = SimulatedActions()
    if function_name in actions:
        func = actions[function_name]
        return func(**params)
    else:
        raise ValueError(f"Function {function_name} not found.")


class SimulatedActions:
    def __init__(self):
        self.actions = get_autog_actions()

    def __getitem__(self, key):
        return self.actions[key]

    def __iter__(self):
        return iter(self.actions.items())

    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        return f"SimulatedActions({self.actions})"


def get_autog_actions():
    G_DUMMY_T = generate_or_connect_dummy_table
    C_TWO_C = connect_two_columns
    E_MULTI_C = explode_multi_category_column
    G_NON_DUMMY_T = generate_non_dummy_table
    RM_PK = remove_primary_key
    ADD_PK = add_primary_key

    actions = {
        "generate_or_connect_dummy_table": G_DUMMY_T,
        "connect_two_columns": C_TWO_C,
        "explode_multi_category_column": E_MULTI_C,
        "generate_non_dummy_table": G_NON_DUMMY_T,
        "remove_primary_key": RM_PK,
        "add_primary_key": ADD_PK,
    }
    return actions


def pack_function_introduction_prompt(func):
    doc_string = inspect.getdoc(func)
    parameters = inspect.signature(func).parameters
    intro = f"Here is the introduction of the function {func.__name__}:\n"
    intro += doc_string
    # intro += "The function has the following parameters:\n"
    # for param in parameters:
    #     intro += f"{param}: {parameters[param]}\n"
    return intro


## here we generate code, the first return for schema, the second return for data


def remove_primary_key(base_table_name, col_name):
    """
    Description:
    Remove a primary key constraint from a column in the original table
    If the column is just an index, then the column will be removed from the table
    Parameters:
    base_table_name: the name of the original table
    col_name: the name of the column in the original table
    """
    schema_code = f"""
tables=schema['tables']
new_tables = []
for table in tables:
    if table['name'] == '{base_table_name}':
        columns = []
        for column in table['columns']:
            if column['name'] == '{col_name}':
                if column['dtype'] == 'primary_key':
                    continue
            else:
                columns.append(column)
        new_tables.append(table)
    else:
        new_tables.append(table)
    """
    data_code = ""
    return schema_code, data_code


def add_primary_key(base_table_name, col_name):
    """
    Description:
    Add a primary key column to the original table
    Parameters:
    base_table_name: the name of the original table
    col_name: the name of the column in the original table
    """
    schema_code = f"""
tables=schema['tables']
for table in tables:
    if table['name'] == '{base_table_name}':
        table['columns'].append({{
            "name": "{col_name}",
            "dtype": "primary_key"
        }})
        break
    """

    data_code = f"""
import pandas as pd
import numpy as np
tables=schema['tables']
target_table = [table for table in tables if table['name'] == '{base_table_name}'][0]
file_path = target_table['source']
if file_path.endswith('.pqt'):
    data = pd.read_parquet(file_path)
    ndata = data.shape[0]
elif file_path.endswith('.npz'):
    data = np.load(file_path, allow_pickle=True)
    data = dict(data)
    first_key = list(data.keys())[0]
    ndata = len(data[first_key])
new_column = np.arange(ndata)
data['{col_name}'] = new_column
if file_path.endswith('.pqt'):
    data.to_parquet(file_path)
else:
    np.savez(file_path, **data)
    """

    return schema_code, data_code


def generate_or_connect_dummy_table(base_table_name, orig_col_name, new_table_name, new_col_name):
    """
    Description:
    This function can be used in two ways:
    1. Generate a dummy table with only one primary key
    2. Turn an existing column with categorical type to an existing dummy table
    "orig_col_name" must be a column with category type
    Parameters:
    base_table_name: the name of the original table
    orig_col_name: the name of the original column in the original table, this should be a column with category type
    new_table_name: the name of the new table to be created/connected
    new_col_name: the name of the new column to be created/connected
    """
    schema_code = f"""
tables=schema['tables']
for i, table in enumerate(tables):
    if table['name'] == '{base_table_name}':
        for j, column in enumerate(table['columns']):
            if column['name'] == '{orig_col_name}':
                if column['dtype'] != 'category': raise ValueError("The column must be a category type")
                table['columns'][j]['dtype'] = 'foreign_key'
                table['columns'][j]['link_to'] = '{new_table_name}.{new_col_name}'
                break
        tables[i] = table
        break
schema['tables'] = tables
    """
    data_code = """"""
    return schema_code, data_code


def connect_two_columns(
    table_1_name,
    table_1_col_name,
    table_2_name,
    table_2_col_name,
    new_table_name="",
    new_table_col_name="",
):
    """
    Description:
    Connect two columns, this function can be used for the following case. Always put the column with category type in table 1.
    1. A category column in table 1 is connected to a category column in table 2, in this case, a new dummy table will be created
    2. A category column in table 1 is connected to a primary key column in table 2, in this case, the column in table 1 will be turned into a foreign key column. In case 2, table_2_col_name must be a primary key column
    3. A category column in table 1 is connected to a non-category and non-primary key column in table 2, in this case, we will use a trick called Surrogate Key.
    Parameters:
    table_1_name: the name of the first table,
    table_1_col_name: the name of the column in the first table, this should be a column with category type
    table_2_name: the name of the second table
    table_2_col_name: the name of the column in the second table, this should be a column with category type
    new_table_name: the name of the new table to be created, can leave it empty for case 2 and 3
    new_table_col_name: the name of the new column to be created, can leave it empty for case 2 and 3
    """
    schema_code = f"""
tables=schema['tables']

## find the primary key for table 2
pk = [column for table in tables if table['name'] == '{table_2_name}' for column in table['columns'] if column['dtype'] == 'primary_key']
add_fk = False
table_1_idx = 0
table_2_idx = 0
for i, table in enumerate(tables):
    if table['name'] == '{table_1_name}':
        for j, column in enumerate(table['columns']):
            if column['name'] == '{table_1_col_name}':
                if column['dtype'] != 'category': raise ValueError("The column must be a category type")
                table['columns'][j]['dtype'] = 'foreign_key'
                if new_table_name == "":
                    table['columns'][j]['link_to'] = '{table_2_name}.{table_2_col_name}'
                else:
                    table['columns'][j]['link_to'] = '{new_table_name}.{new_table_col_name}'
                break
        table_1_idx = i
        tables[i] = table
    elif table['name'] == '{table_2_name}':
        add_pk = False
        for j, column in enumerate(table['columns']):
            if column['name'] == '{table_2_col_name}':
                if column['dtype'] != 'category' and new_table_name != '': raise ValueError("The column must be a category type")
                elif column['dtype'] == 'category' and new_table_name == '': raise ValueError("In this case you must provide a new table name")
                elif column['dtype'] != 'category' and new_table_name == '':
                    ## Surrogate Key
                    if len(pk) == 0:
                        ## add a primary key for table 2
                        add_pk = True
                    add_fk = True
                else:
                    table['columns'][j]['dtype'] = 'foreign_key'
                    table['columns'][j]['link_to'] = '{new_table_name}.{new_table_col_name}'
                    break
        if add_pk:
            table['columns'].append({{
                "name": "{table_2_col_name}ID",
                "dtype": "primary_key"
            }})
            ## open table 2
            format = table['format']
            source = table['source']
            if format == 'parquet':
                data = pd.read_parquet(source)
                data['{table_2_col_name}ID'] = np.arange(data.shape[0])
                data.to_parquet(source)
            else:
                data = np.load(source, allow_pickle=True)
                data = dict(data)
                data['{table_2_col_name}ID'] = np.arange(len(data['{table_2_col_name}']))
                np.savez(source, **data)
        table_2_idx = i
        tables[i] = table
if add_fk:
    ## mapping between table_2_col_name and pk of table 2
    table2_pk_name = pk[0]['name'] if len(pk) > 0 else "{table_2_col_name}ID"
    ## change the schema
    for i, table in enumerate(tables):
        if table['name'] == '{table_1_name}':
            for j, column in enumerate(table['columns']):
                if column['name'] == '{table_1_col_name}':
                    table['columns'][j]['dtype'] = 'foreign_key'
                    table['columns'][j]['link_to'] = f'{table_2_name}.{{table2_pk_name}}'
                    break
    table2 = tables[table_2_idx]
    ## open table 2
    format = table2['format']
    source = table2['source']
    if format == 'parquet':
        data = pd.read_parquet(source)
    else:
        data = np.load(source, allow_pickle=True)
        data = dict(data)
    ## mapping between table_2_col_name and pk of table 2
    table2_pk_name = pk[0]['name'] if len(pk) > 0 else "{table_2_col_name}ID"
    table2_pk = data[table2_pk_name]
    table2_col = data['{table_2_col_name}']
    mapping = {{k:v for k, v in zip(table2_col, table2_pk)}}
    ## map into table 1
    table1 = tables[table_1_idx]
    format = table1['format']
    source = table1['source']
    if format == 'parquet':
        data = pd.read_parquet(source)
        data['{table_1_col_name}'] = data['{table_1_col_name}'].map(mapping)
        data.to_parquet(source)
    else:
        data = np.load(source, allow_pickle=True)
        data = dict(data)
        data['{table_1_col_name}'] = [mapping[val] for val in data['{table_1_col_name}']]
        np.savez(source, **data)
schema['tables'] = tables
        """
    data_code = """
        """
    return schema_code, data_code


def explode_multi_category_column(
    original_table, multi_cat_col, primary_key_column, new_table_name, new_col_name, dtype
):
    """
    Description:
    Explode a multi-category column into multiple columns
    Parameters:
    original_table: name of the original table where the multi-category column is located
    multi_cat_col: the name of the multi-category column
    primary_key_column: the name of the primary key column in the original table
    new_table_name: the name of the new table to be created
    new_col_name: the name of the new column to be created
    dtype: the data type of the new column, if set to "foreign_key", this table will contain only "foreign_keys". In this case, it means you only want to use this column's relaion. If set to other types, this table will contain the original column's values, and a primary key will be added, this means you want to use this column's values.
    """
    if dtype != "foreign_key":
        schema_code = f"""
tables=schema['tables']
del_col_idx = 0
del_table_idx = 0
for i, table in enumerate(tables):
    if table['name'] == '{original_table}':
        for j, column in enumerate(table['columns']):
            if column['name'] == '{multi_cat_col}':
                del_col_idx = j
        del_table_idx = i
tables[del_table_idx]['columns'].pop(del_col_idx)
new_table = {{
    "name": "{new_table_name}",
    "columns": [
        {{
            "name": "{primary_key_column}",
            "dtype": "foreign_key",
            "link_to": "{original_table}.{primary_key_column}"
        }},
        {{
            "name": "{new_col_name}",
            "dtype": "{dtype}"
        }},
        {{
            "name": "{new_table_name}ID",
            "dtype": "primary_key"
        }}
    ]
}}
tables.append(new_table)
schema['tables'] = tables
        """
    else:
        schema_code = f"""
tables=schema['tables']
for i, table in enumerate(tables):
    if table['name'] == '{original_table}':
        for j, column in enumerate(table['columns']):
            if column['name'] == '{multi_cat_col}':
                table['columns'][j]['dtype'] = 'foreign_key'
                table['columns'][j]['link_to'] = '{new_table_name}.{new_col_name}'
                break
        tables[i] = table
new_table = {{
    "name": "{new_table_name}",
    "format": "parquet",
    "source": "data/{new_table_name}.pqt",
    "columns": [
        {{
            "name": "{primary_key_column}",
            "dtype": "foreign_key",
            "link_to": "{original_table}.{primary_key_column}"
        }},
        {{
            "name": "{new_col_name}",
            "dtype": "foreign_key",
            "link_to": "{new_col_name}.{new_col_name}ID"
        }}
    ]
}}
tables.append(new_table)
schema['tables'] = tables
        """
    data_code = f"""
import pandas as pd
import numpy as np
tables=schema['tables']
target_table = [table for table in tables if table['name'] == '{original_table}'][0]
file_path = target_table['source']
if file_path.endswith('.pqt'):
    data = pd.read_parquet(file_path)
    new_df = data[['{primary_key_column}', '{multi_cat_col}']].explode('{multi_cat_col}').reset_index(drop=True)
    new_df = new_df.rename(columns={{'{multi_cat_col}': '{new_col_name}'}})
    if dtype != 'foreign_key':
        new_df['{new_table_name}ID'] = np.arange(new_df.shape[0])
    new_file_path = "data/{new_table_name}.pqt"
    data = data.drop('{multi_cat_col}', axis=1)
    new_df.to_parquet(new_file_path)
    data.to_parquet(file_path)
else:
    data = np.load(file_path, allow_pickle=True)
    data = dict(data)
    rel_cols = {{k:v for k, v in data.items() if k in ['{primary_key_column}', '{multi_cat_col}']}}
    df = pd.DataFrame(rel_cols)
    del data['{multi_cat_col}']
    np.savez(file_path, **data)
    new_df = df[['{primary_key_column}', '{multi_cat_col}']].explode('{multi_cat_col}').reset_index(drop=True)
    new_df = new_df.rename(columns={{'{multi_cat_col}': '{new_col_name}'}})
    if dtype != 'foreign_key':
        new_df['{new_table_name}ID'] = np.arange(new_df.shape[0])
    new_file_path = "data/{new_table_name}.pqt"
    new_df.to_parquet(new_file_path)
    """
    return schema_code, data_code


def generate_non_dummy_table(base_table_name, cols, new_table_name):
    """
    Description:
    Generate a non-dummy table with columns in the original table
    Parameters:
    base_table_name: the name of the original table
    cols: the list of columns to be included in the new table and removed from the original table
    new_table_name: the name of the new table to be created
    """
    schema_code = f"""
tables=schema['tables']
del_idx = []
col_info = []
target_table_idx = 0
for i, table in enumerate(tables):
    if table['name'] == '{base_table_name}':
        for j, column in enumerate(table['columns']):
            if column['name'] in {cols}:
                del_idx.append((i, column['name']))
                col_info.append(column)
                target_table_idx = i
tables[target_table_idx]['columns'].append({{
    "name": "{new_table_name}ID",
    "dtype": "foreign_key",
    "link_to": "{new_table_name}.{new_table_name}ID"
}})


for (x, y) in del_idx:
    new_cols = [col for col in tables[x]['columns'] if col['name'] != y]
    tables[x]['columns'] = new_cols
col_info.append({{
    "name": "{new_table_name}ID",
    "dtype": "primary_key"
}})
tables.append({{
    "name": "{new_table_name}",
    "columns": col_info,
    "source": "data/{new_table_name}.pqt",
    "format": "parquet"
}})
schema['tables'] = tables
    """

    data_code = f"""
import pandas as pd
import numpy as np
tables=schema['tables']
target_table = [table for table in tables if table['name'] == '{base_table_name}'][0]
file_path = target_table['source']
if file_path.endswith('.pqt'):
    data = pd.read_parquet(file_path)
    ndata = data[{cols}]
    data = data.to_dict()
elif file_path.endswith('.npz'):
    data = np.load(file_path, allow_pickle=True)
    data = dict(data)
    ndata = {{k:v for k, v in data.items() if k in {cols}}}
    ndata = pd.DataFrame(ndata)
new_file_path = "data/{new_table_name}.pqt"
df_new = ndata.drop_duplicates().reset_index(drop=True).reset_index()
df_new.rename(columns={{'index': '{new_table_name}ID'}}, inplace=True)
df_new.to_parquet(new_file_path)
ndata = ndata.merge(df_new, on={cols}, how='left')
data = {{k:v for k, v in data.items() if k not in {cols}}}
if file_path.endswith('.pqt'):
    data['{foreign_key_col}'] = ndata['{new_table_name}ID']
    data = pd.DataFrame(data)
    data.to_parquet(file_path)
else:
    data['{foreign_key_col}'] = ndata['{new_table_name}ID'].values
    np.savez(file_path, **data)
    """

    return schema_code, data_code
