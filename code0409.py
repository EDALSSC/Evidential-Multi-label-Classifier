# # 1. 导入库
# import pandas as pd
# import numpy as np
# import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 2. 读取数据
# df = pd.read_excel("did_tourism_stock_large_dataset.xlsx")
#
# # 3. 可选：时间格式转换
# df["date"] = pd.to_datetime(df["date"])
#
# # 4. 平行趋势图
# df_plot = df.groupby(["date", "treat"])["return"].mean().reset_index()
# df_plot["group"] = df_plot["treat"].map({1: "旅游行业", 0: "非旅游行业"})
#
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df_plot, x="date", y="return", hue="group", marker="o")
# plt.axvline(pd.to_datetime("2023-07-01"), color="gray", linestyle="--", label="政策拐点")
# plt.title("平行趋势图：旅游 vs 非旅游行业")
# plt.xlabel("时间")
# plt.ylabel("平均收益率")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 5. 构建DID变量（如未存在）
# df["did"] = df["treat"] * df["post"]
# df["time_index"] = (df["date"] - df["date"].min()).dt.days
#
# # 6. 回归模型
# model_did = smf.ols("return ~ treat + post + did", data=df).fit()
# model_trend = smf.ols("return ~ treat + post + did + time_index + treat:time_index", data=df).fit()
# model_robust = smf.ols("return ~ treat + post + did", data=df).fit(cov_type="HC3")
# model_fe = smf.ols("return ~ treat + post + did + C(stock_id)", data=df).fit()
#
# print("DID标准模型：\n", model_did.summary())
# print("加入趋势项模型：\n", model_trend.summary())
# print("异方差稳健模型：\n", model_robust.summary())
# print("固定效应模型：\n", model_fe.summary())
#
# # 7. 可视化：DID主效应置信区间图
# ci = model_did.conf_int()
# ci.columns = ["lower", "upper"]
# ci["coef"] = model_did.params
# ci = ci.loc[["treat", "post", "did"]]
#
# ci.plot(kind="barh", xerr=(ci["upper"] - ci["lower"]) / 2, figsize=(8, 4), legend=False)
# plt.axvline(0, color="black", linestyle="--")
# plt.title("DID主效应置信区间")
# plt.xlabel("系数")
# plt.tight_layout()
# plt.show()
#
# # 8. 导出结果为Excel
# def summary_to_df(model):
#     return pd.DataFrame({
#         "coef": model.params,
#         "std err": model.bse,
#         "t": model.tvalues,
#         "P>|t|": model.pvalues
#     })
#
# with pd.ExcelWriter("did_results_summary.xlsx") as writer:
#     summary_to_df(model_did).to_excel(writer, sheet_name="Standard DID")
#     summary_to_df(model_trend).to_excel(writer, sheet_name="Trend")
#     summary_to_df(model_robust).to_excel(writer, sheet_name="Robust")
#     summary_to_df(model_fe).to_excel(writer, sheet_name="Fixed Effects")
# 1. 导入库
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 设置字体为支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 2. 读取数据
df = pd.read_excel("did_tourism_stock_large_dataset.xlsx")

# 3. 可选：时间格式转换
df["date"] = pd.to_datetime(df["date"])

# 4. 平行趋势图
df_plot = df.groupby(["date", "treat"])["return"].mean().reset_index()
df_plot["group"] = df_plot["treat"].map({1: "旅游行业", 0: "非旅游行业"})

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_plot, x="date", y="return", hue="group", marker="o")
plt.axvline(pd.to_datetime("2023-07-01"), color="gray", linestyle="--", label="政策拐点")
plt.title("平行趋势图：旅游 vs 非旅游行业")
plt.xlabel("时间")
plt.ylabel("平均收益率")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 构建DID变量（如未存在）
df["did"] = df["treat"] * df["post"]
df["time_index"] = (df["date"] - df["date"].min()).dt.days

# 6. 回归模型
# 标准DID模型
model_did = smf.ols("return ~ treat + post + did", data=df).fit()

# 加入趋势项模型
model_trend = smf.ols("return ~ treat + post + did + time_index + treat:time_index", data=df).fit()

# 异方差稳健模型
model_robust = smf.ols("return ~ treat + post + did", data=df).fit(cov_type="HC3")

# 固定效应模型
model_fe = smf.ols("return ~ treat + post + did + C(stock_id)", data=df).fit()

# 打印回归结果
print("DID标准模型：\n", model_did.summary())
print("加入趋势项模型：\n", model_trend.summary())
print("异方差稳健模型：\n", model_robust.summary())
print("固定效应模型：\n", model_fe.summary())

# 7. 可视化：DID主效应置信区间图
ci = model_did.conf_int()
ci.columns = ["lower", "upper"]
ci["coef"] = model_did.params
ci = ci.loc[["treat", "post", "did"]]

ci.plot(kind="barh", xerr=(ci["upper"] - ci["lower"]) / 2, figsize=(8, 4), legend=False)
plt.axvline(0, color="black", linestyle="--")
plt.title("DID主效应置信区间")
plt.xlabel("系数")
plt.tight_layout()
plt.show()

# 8. 导出结果为Excel
def summary_to_df(model):
    return pd.DataFrame({
        "coef": model.params,
        "std err": model.bse,
        "t": model.tvalues,
        "P>|t|": model.pvalues
    })

with pd.ExcelWriter("did_results_summary.xlsx") as writer:
    summary_to_df(model_did).to_excel(writer, sheet_name="Standard DID")
    summary_to_df(model_trend).to_excel(writer, sheet_name="Trend")
    summary_to_df(model_robust).to_excel(writer, sheet_name="Robust")
    summary_to_df(model_fe).to_excel(writer, sheet_name="Fixed Effects")
