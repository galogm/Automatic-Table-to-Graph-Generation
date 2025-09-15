import json

task_description = {
    "mag": {
        "venue": "This task is to predict the venue of a paper in target_column `{target_column}` of target_table `{target_table}`  given the paper's title, abstract, authors, and publication year. \
        You may use the meta relations between papers, authors, topics, and institutions to improve the performance",
        "year": "This task is to predict the publication year of a paper in target_column `{target_column}` of target_table `{target_table}` given the paper's title, abstract, authors, and venue. \
        You may use the meta relations between papers, authors, topics, and institutions to improve the performance",
        "cite": "This task is to predict whether two papers are cited by each other in target_column `{target_column}` of target_table `{target_table}`  given the paper's title, abstract, authors, and venue. \
            You may use the meta relations between papers, authors, topics, and institutions to improve the performance",
    },
    "movielens": {
        "ratings": "This task is to predict user's ratings on movies in target_column `{target_column}` of target_table `{target_table}` given movie information and movie-user structural information."
    },
    "avs": {
        "repeater": "This task is to predict whether a user will repeat a purchase in target_column `{target_column}` of target_table `{target_table}`  given the user's purchase history and user-item structural information"
    },
    "ieeecis": {
        "fraud": "This task is to predict whether a transaction is fraudulent in target_column `{target_column}` of target_table `{target_table}`  given the transaction information and user-transaction structural information"
    },
    "diginetica": {
        "ctr": "This task is to predict the click-through rate of an ad in target_column `{target_column}` of target_table `{target_table}`  given the ad information and user-ad structural information. In the task table, you are given itemId, queryId, timestamp, and clicked. The target is clicked. Moreover, itemId is a foreign key pointing to itemId of the Product table. queryId is a foreign key pointing to queryId of the Query table.",
        "purchase": "This task is to predict whether a user will purchase an item in target_column `{target_column}` of target_table `{target_table}` given the item information and user-item structural information. In the task table, you are given itemId, queryId, timestamp, and clicked. The target is clicked. Moreover, itemId is a foreign key pointing to itemId of the Product table. purchase_session is a foreign key pointing to the Session table, which inspires that there should be one table Session",
    },
    "retailrocket": {
        "cvr": "The task is to classify whether an item will be added to the shopping cart by a visitor, i.e. predicting column View.added_to_cart, in target_column `{target_column}` of target_table `{target_table}`."
    },
    "outbrain": {
        "ctr": "The task is to predict whether a promoted content will be clicked or not, i.e. predicting Click.clicked, in target_column `{target_column}` of target_table `{target_table}`."
    },
    "stackexchange": {
        "upvote": "The task is to predict the Target column of table Posts, which means predicting whether the post will be upvoted or not, in target_column `{target_column}` of target_table `{target_table}` .",
        "churn": "The task is to predict the Target column of table Users, which means predicting whether the user will churn or not,  in target_column `{target_column}` of target_table `{target_table}` .",
    },
}


def get_task_description(
    dataset: str, task_name: str, target_column: str, target_table: str
):
    try:
        return f"{task_description[dataset][task_name].format(target_column=target_column, target_table=target_table)}. The starting and final columns should both be the column `{target_column}` of the table `{target_table}`"
    except KeyError:
        return ""


def get_task_meta_info(schema, selected_task):
    task_info = schema["tasks"]
    meta_str = ""
    sel_task = None
    for i, info in enumerate(task_info):
        if info["name"] == selected_task:
            # import ipdb; ipdb.set_trace()
            meta_str += json.dumps(info)
            sel_task = info
            break
    meta_str += "\n"
    meta_str += f"Our target is to predict {sel_task['target_table']}.{sel_task['target_column']}. Don't change it."
    return meta_str


def get_task_meta(schema, selected_task):
    task_info = schema["tasks"]
    for i, info in enumerate(task_info):
        if info["name"] == selected_task:
            return info
