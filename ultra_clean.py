import pandas as pd
import re
import yaml

def load_dict(yaml_path="dict.yaml"):
    """读取 dict.yaml 并返回原始字典和合并词列表"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    terms = []
    for category, words in d.items():
        if words:
            terms.extend(words)
    return d, terms

def contains_terms(text: str, terms: list) -> bool:
    """检查句子是否包含任意词典词"""
    text = str(text)
    return any(term in text for term in terms)

def contains_latin(text: str) -> bool:
    """检查句子是否含有拉丁字母"""
    text = str(text)
    return bool(re.search(r"[A-Za-z]", text))

def count_category_hits(df, category_dict):
    """统计每个类别被匹配的句子数量"""
    counts = {}
    for category, words in category_dict.items():
        if not words:
            counts[category] = 0
            continue
        counts[category] = df.apply(
            lambda row: any(term in str(row["sentence1"]) or term in str(row["sentence2"]) for term in words),
            axis=1
        ).sum()
    return counts

def main():
    # 读取原始中文数据
    df = pd.read_csv("data/xnli_zh_dev.csv")

    # 先删除含拉丁字母的句子
    df["has_latin"] = df.apply(
        lambda row: contains_latin(row["sentence1"]) or contains_latin(row["sentence2"]),
        axis=1
    )
    df_no_latin = df[df["has_latin"] == False]

    # 加载人工词典
    category_dict, all_terms = load_dict("dict.yaml")

    # 标记是否含有词典中的词
    df_no_latin["has_foreign_term"] = df_no_latin.apply(
        lambda row: contains_terms(row["sentence1"], all_terms) or 
                    contains_terms(row["sentence2"], all_terms),
        axis=1
    )

    # clean group = 不含拉丁字母 且 不含词典词
    clean_df = df_no_latin[df_no_latin["has_foreign_term"] == False]

    # problem group = 含拉丁字母 或含词典词
    problem_df = df[ df["has_latin"] | df_no_latin["has_foreign_term"] ]  # 注意逻辑

    # 保存 clean dataset
    clean_df.to_csv("data/xnli_zh_clean_dev.csv", index=False, encoding="utf-8")

    # 保存 problem dataset
    problem_df.to_csv("data/xnli_zh_problem_dev.csv", index=False, encoding="utf-8")

    # 统计每个类别被删除的句子数量（基于去掉拉丁字母后的数据 df_no_latin）
    category_counts = count_category_hits(df_no_latin, category_dict)

    print(f"原始数据: {len(df)} 条")
    print(f"去掉拉丁字母后: {len(df_no_latin)} 条")
    print(f"最终清洁组: {len(clean_df)} 条")
    print(f"问题组: {len(problem_df)} 条")
    print(f"删除的句子统计（按类别, 基于拉丁字母过滤后的数据）:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} 条")

if __name__ == "__main__":
    main()


# # save only cleaned
# import pandas as pd
# import re
# import yaml

# def load_dict(yaml_path="dict.yaml"):
#     """读取 dict.yaml 并返回原始字典和合并词列表"""
#     with open(yaml_path, "r", encoding="utf-8") as f:
#         d = yaml.safe_load(f)
#     terms = []
#     for category, words in d.items():
#         if words:
#             terms.extend(words)
#     return d, terms

# def contains_terms(text: str, terms: list) -> bool:
#     """检查句子是否包含任意词典词"""
#     text = str(text)
#     return any(term in text for term in terms)

# def contains_latin(text: str) -> bool:
#     """检查句子是否含有拉丁字母"""
#     text = str(text)
#     return bool(re.search(r"[A-Za-z]", text))

# def count_category_hits(df, category_dict):
#     """统计每个类别被匹配的句子数量"""
#     counts = {}
#     for category, words in category_dict.items():
#         if not words:
#             counts[category] = 0
#             continue
#         counts[category] = df.apply(
#             lambda row: any(term in str(row["sentence1"]) or term in str(row["sentence2"]) for term in words),
#             axis=1
#         ).sum()
#     return counts

# def main():
#     # 读取原始中文数据
#     df = pd.read_csv("data/xnli_zh_dev.csv")

#     # 先删除含拉丁字母的句子
#     df["has_latin"] = df.apply(
#         lambda row: contains_latin(row["sentence1"]) or contains_latin(row["sentence2"]),
#         axis=1
#     )
#     df_no_latin = df[df["has_latin"] == False]

#     # 加载人工词典
#     category_dict, all_terms = load_dict("dict.yaml")

#     # 标记是否含有词典中的词
#     df_no_latin["has_foreign_term"] = df_no_latin.apply(
#         lambda row: contains_terms(row["sentence1"], all_terms) or 
#                     contains_terms(row["sentence2"], all_terms),
#         axis=1
#     )

#     # clean group = 不含拉丁字母 且 不含词典词
#     clean_df = df_no_latin[df_no_latin["has_foreign_term"] == False]

#     # 保存 clean dataset
#     clean_df.to_csv("data/xnli_zh_clean_dev.csv", index=False, encoding="utf-8")

#     # 统计每个类别被删除的句子数量（基于原始去掉拉丁字母后的数据 df_no_latin）
#     category_counts = count_category_hits(df_no_latin, category_dict)

#     print(f"原始数据: {len(df)} 条")
#     print(f"去掉拉丁字母后: {len(df_no_latin)} 条")
#     print(f"最终清洁组: {len(clean_df)} 条")
#     print(f"删除的句子统计（按类别, 基于拉丁字母过滤后的数据）:")
#     for cat, count in category_counts.items():
#         print(f"  {cat}: {count} 条")

# if __name__ == "__main__":
#     main()


# ## Only delete dict
# # import pandas as pd
# # import yaml

# # def load_dict(yaml_path="dict.yaml"):
# #     """读取 dict.yaml 并合并所有类别为一个 list"""
# #     with open(yaml_path, "r", encoding="utf-8") as f:
# #         d = yaml.safe_load(f)
# #     terms = []
# #     for category, words in d.items():
# #         if words:  # 防止空类别
# #             terms.extend(words)
# #     return terms

# # def contains_foreign_terms(text: str, terms: list) -> bool:
# #     """检查句子中是否包含人工词典中的词"""
# #     text = str(text)
# #     return any(term in text for term in terms)

# # def main():
# #     # 读取已经去掉拉丁字母的中文数据
# #     df = pd.read_csv("data/xnli_zh_dev.csv")

# #     # 加载人工词典
# #     foreign_terms = load_dict("dict.yaml")

# #     # 标记是否含有词典中的词
# #     df["has_foreign_term"] = df.apply(
# #         lambda row: contains_foreign_terms(row["sentence1"], foreign_terms) or 
# #                     contains_foreign_terms(row["sentence2"], foreign_terms),
# #         axis=1
# #     )

# #     # clean group = 不含任何词典词语
# #     clean_df = df[df["has_foreign_term"] == False]

# #     # 保存
# #     clean_df.to_csv("data/xnli_zh_clean_dev.csv", index=False, encoding="utf-8")

# #     print(f"总数据: {len(df)} 条")
# #     print(f"清洁组: {len(clean_df)} 条 (已删除包含 dict.yaml 中词语的句子)")

# # if __name__ == "__main__":
# #     main()
