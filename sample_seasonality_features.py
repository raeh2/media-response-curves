def make_date(df: pd.DataFrame, date_field: str):
    "Make sure `df[field_name]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(
    df: pd.DataFrame,
    field_name: str = None,
    prefix: str = "Dates__FE__",
    drop: bool = True,
    index_field: bool = False,
):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    if index_field:
        field_name = df.index.name
        drop = False
        df.reset_index(inplace=True)

    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub("[Dd]ate$", "", field_name))
    attr = ["Year", "Month", "Week", "Day"]
    for n in attr:
        dt = "category"
        if (n == "Week") | (n == "Day"):
            dt = "int64"
        df[prefix + n] = getattr(field.dt, n.lower()).astype(dt)

    df[prefix + "Elapsed"] = field.astype(np.int64) // 10 ** 9
    df[prefix + "WeekOfMonth"] = ((df[prefix + "Day"] - 1) // 7 + 1).astype("category")
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    if index_field:
        df = df.set_index(field_name, inplace=True)

    return df