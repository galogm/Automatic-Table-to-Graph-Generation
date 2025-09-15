def get_autom_action_selection_prompt(history_actions, input_schema, stats, task):
    MULTI_ROUND = """You are an excellent expert **graph data scientist and engineer**. Your task is to discover as many **meta-relations** as possible for constructing **task-oriented graphs** based on the given input Relational Database schema and statistics.

The schema will be provided in a dictionary format as follows:

```yaml
<data>
1. dataset_name: name of the dataset  
2. tables: list of tables, each containing:  
   - name: table name  
   - source: source of the data (either a `.npz` or `.parquet` file)  
   - columns: list of columns, where each column includes:  
       * name: column name  
       * dtype: column type (choices: text, categorical, float, primary_key, foreign_key, multi_category)  
           - `primary_key` and `foreign_key` are special categorical types defining structural relations across tables.  
           - `multi_category` indicates that a column contains lists of categorical values.  
           - Once a column is designated as `primary_key` or `foreign_key`, its type cannot be changed.  
       * link_to (optional): if the column is a `foreign_key`, this specifies the target table and primary key it links to.  
3. statistics: descriptive statistics of table columns, which can be used to infer column characteristics.  
   - For example, if a categorical column has only one unique value, creating a node type from it may lead to a supernode, which is undesirable.  
   - Statistics also help in determining whether two columns represent the same concept.  
</data>
```

### Function Documentation: `generate_meta_relation_table`

**Description**
This function supports constructing meta-relations in two modes:

1. **Relation chaining across columns**

   * Defines a connection between rows across multiple tables by traversing attribute paths.
   * For example:

     ```py
     generate_meta_relation_table(dbb, 
                                  table_1: str, table_1_source_cols_1: List[str], table_1_target_cols_2: List[str], 
                                  table_2: str, table_2_source_cols_1: List[str], table_2_target_cols_2: List[str], 
                                  ...,
                                  table_n: str, table_n_source_cols_1: List[str], table_n_target_cols_2: List[str])
     ```

     means:

     * From `table_1`, map elements in `table_1_cols_1` to related elements in `table_1_cols_2`.
     * Use `table_1_cols_2` to locate corresponding values in `table_2_cols_1`.
     * Retrieve the related values in `table_2_cols_2`.
     * Continue this chaining process until reaching `table_n`.

2. **Head–tail consistency constraint**

   * The head columns (`table_1_cols_1`) must be the **target columns in the target table of the task**, ensuring task-oriented meta-relations.
   * The head column (`table_1_cols_1`) and the tail column (`table_n_cols_2`) must belong to the **same columns of the same table**, ensuring meta-relation closure.

**Parameters**

* `dbb`: the database object
* `table_1`: name of the first table
* `table_1_cols_1`: head column of `table_1` (task-relevant)
* `table_1_cols_2`: related column in `table_1`
* `table_2`: name of the second table (can be the same or different)
* `table_2_cols_1`: column in `table_2` that directly maps to `table_1_cols_2` (via shared FK–PK relation or semantic alignment)
* `table_2_cols_2`: task-relevant column in `table_2`
* … continue for `table_3` to `table_n`
* `table_n_cols_2`: must equal `table_1_cols_1` to close the meta-relation

### Your Task

1. Analyze the given `<dataset_stats>`, `<schema>`, and `<tasks>` blocks.
2. Decide whether constructing a **meta-relation** is beneficial for the task.
3. If yes, propose a **meta-relation** by specifying the function call to `generate_meta_relation_table`.
4. Provide your reasoning in natural language (`explanation`).
5. Provide each meta-relation a corresponding sql (`sqls`). If there is any timestamp in the meta-relation path, keep the latest in the final table searched out by the sql. No row of a later timestamp can be joined by an early row.
6. Keep all rows distinct, removing any duplicates. For edges that are invertible (i.e., both (A, B) and (B, A) exist), retain only one of them.
7. Provide each table generated from the meta-relation a corresponding schema (`new_table`) in the format of given original schema as follows. Keep the names the same as those in the sql code.
```
{{
    "name": "Ratings",
    "format": "parquet",
    "columns": [
        {{
            "name": "rate_user",
            "dtype": "category",
        }},
        {{
            "name": "rate_movie",
            "dtype": "foreign_key",
            "link_to": "Movies.movieID"
        }},
        {{
            "name": "rating",
            "dtype": "category",
        }},
        {{
            "name": "timestamp",
            "dtype": "datetime",
        }},
        {{
            "name": "ratingID",
            "dtype": "primary_key",
        }}
    ],
    "time_column": "timestamp"
}}
```

### Expected Output Format

Return your output as JSON inside `<selection></selection>`. Example:

```xml
<selection>
[
  {{
   "explanation": "Users who have rated the same movies often share similar preferences. By constructing a meta-relation that connects a rating to other ratings on the same movie, we can capture collaborative filtering signals for predicting a user's rating.",
    "action": "generate_meta_relation_table",
    "parameters": {{
            "table_1": "Ratings",
            "table_1_cols_1": [
                "rating"
            ],
            "table_1_cols_2": [
                "rate_movie"
            ],
            "table_2": "Ratings",
            "table_2_cols_1": [
                "rate_movie"
            ],
            "table_2_cols_2": [
                "rating"
            ]
        }},
    "sqls": [
            "SELECT DISTINCT r1.rating AS head_rating, r2.rating AS tail_rating FROM Ratings r1 JOIN Ratings r2 ON r1.rate_movie = r2.rate_movie WHERE r1.ratingID < r2.ratingID;"
        ]
    "new_table": {{
                "name": "Category_PaperID_Author_Journal_PaperID_Category",
                "source": "data/Category_PaperID_Author_Journal_PaperID_Category.npz",
                "columns": [
                    {{"name": "head_rating_table1", "dtype": "foreign_key", "link_to": "Ratings.rating"}},
                    {{"name": "tail_rating_table1", "dtype": "foreign_key", "link_to": "Ratings.rating"}},
                ]
            }}
  }}
]
</selection>
```

<input>

<dataset_stats> 
    {stats}
</dataset_stats> 

<task> 
    {task}
</task> 

<schema> 
    {input_schema}
</schema>
        
History Actions:
    {history_actions}
</input> 

Return your output in the json format inside <selection></selection>.
"""
    return MULTI_ROUND.format(
        history_actions=history_actions,
        stats=stats,
        task=task,
        input_schema=input_schema,
    )
