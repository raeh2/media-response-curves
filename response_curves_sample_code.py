"""
This is the intellectual property of Raelene Huang (Merkle Inc.)

Confidential. Not to be used for distribution purposes. 
"""


def get_transform_parameters() -> pd.DataFrame:
    func_dict = {
        'log':lambda x: np.log(x + 1),
        'square': lambda x: x**2
    }

    hp_vars = {
        'lag': np.arange(0, 4), #experiment with negative lag 
        'adstock': np.arange(0.5, 1.01, 0.1),
        #'slope': np.arange(0.5, 1.01, 0.1), #changed values after discussion with UK Pioneer Team.
        'slope': np.arange(7000,1_100_000,10000),
        'shape_transform':('square','log','none')}

    hp_grid = pd.DataFrame({'cartesian':1},index=[0])

    for name, ls in hp_vars.items():
        temp_df = pd.DataFrame({name:ls,'cartesian':1})
        hp_grid = hp_grid.merge(temp_df,on='cartesian')

    hp_grid.drop(['cartesian'],axis = 1,inplace=True)

    return hp_vars, hp_grid

# do not touch
def decay(x: pd.Series, decay_val: float, na_val: int = 0) -> pd.Series:
    for i in np.arange(len(x)):
        if i >= 1:
            x.iat[i] = x.iat[i] * decay_val + (1-decay_val) * x.iat[i - 1]
            #x.iat[i] = x.iat[i] + (1-decay_val) * x.iat[i - 1]
    return x

def lag(x: pd.Series, lag_val: int, na_val: int = 0) -> pd.Series:
    x = x.fillna(na_val)
    x = x.shift(lag_val)
    x = x.fillna(na_val)
    return x

def alpha(x: pd.Series, alpha_val: float, slope=True):
    if slope:
        vec = 1 - np.exp(-x / alpha_val)
    else:
        vec = x ** alpha_val
    return vec

def shape_transform(x: pd.Series, shape_transform_val: str = "none", inv: bool = False) -> pd.Series:
    if shape_transform_val == "none":
        return x

    elif shape_transform_val == "log":
        if inv:
            return np.exp(x) - 1
        return np.log(x + 1)

    elif shape_transform_val == "square":
        if inv:
            return np.sqrt(x)
        return x ** 2
    else:
        return x

def apply_transformations(x: pd.Series, adstock_val: float, lag_val: int, slope_val: int, shape_transform_val: str
                         ) -> pd.Series:
    x_new = x.copy()
    x_new = lag(x_new, lag_val)
    x_new = decay(x_new, adstock_val)
    x_new = alpha(x_new, slope_val, slope=True)
    x_new = shape_transform(x_new, shape_transform_val)
    return x_new 

X = pd.DataFrame()
_, hp_grid = get_transform_parameters()
for col in media_columns:
    print("transforming media column: ",col)
    #col="MiQ"
    transformed_features = pd.DataFrame(sales[col])
    for i in hp_grid.iterrows():
        hp_vars = i[1] # all hyperparameter values as series object. i[0] is the index value which is not meaningful
        col_name = "{}:slope{}/adstock{}/lag{}/{}".format(
            col,
            round(hp_vars["slope"],2),
            round(hp_vars["adstock"],2),
            round(hp_vars["lag"],2),
            hp_vars["shape_transform"])
        transformed_features[col_name] = apply_transformations(
            media_plan[col],
            hp_vars["adstock"],
            hp_vars["lag"],
            hp_vars["slope"],
            hp_vars["shape_transform"],
        )
        #print("slope{}".format(round(hp_vars["slope"],3)))
    top = transformed_features.corr()[col].sort_values(ascending=False)[1:5]
    X[top.index] = transformed_features[top.index]
    print(top)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression

# force the curve to stay saturated instead of going down
def response_curve(x: np.array) -> pd.Series:
    for i in np.arange(len(x)):
        if i >= 1:
            if x[i] > x[i-1]:
                x[i] = x[i]
            else:
                x[i] = x[i-1]
    return x

fitted_models = {}
for media in media_columns:
    last_trans = np.nan
    x_media = pd.DataFrame()
    for colnames in X.columns:
        if colnames.startswith(media):
            trans = colnames.strip("/")[-1]
            if trans != last_trans:
                x_media[colnames] = X[colnames]
                last_trans = trans
    
    x_fit = x_media
    y_fit = sales[media]
    
    X_train, X_test, y_train, y_test = train_test_split(x_fit, y_fit,test_size=0.35, random_state=101)
    lm = LinearRegression(fit_intercept=False, normalize=False).fit(X_train,y_train)

    y_pred=lm.predict(X_test)  
    fitted_models[media] = {}
    fitted_models[media]["coef"] = pd.DataFrame(lm.coef_,x_fit.columns,columns=['Coefficient'])
    fitted_models[media]["metric"] = {
                'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                'R-Square': metrics.r2_score(y_test,y_pred)}

    plt.figure(figsize=(25,7))
    plt.plot(y_test, label="original sales")
    plt.plot(y_pred, label="predicted sales")
    plt.legend(loc="upper right")
    plt.title(media + ": R-square "+ str(round(fitted_models[media]["metric"]["R-Square"], 2)))

fitted_models

x = np.arange(1000,9_900_000,1000)
response_curves_output = pd.DataFrame(x,columns=["Spends"])

for media in media_columns:
    uplift = 0
    #original_spends = media_plan[media]
    #original_spends_u = 0
    for colnames in fitted_models[media]["coef"].index:
        trans = colnames.split("/")[-1]
        slope = int(colnames.split("/")[0].split("slope")[-1])

        adstock_ = alpha(x, slope, slope=True)
        uplift = uplift + shape_transform(adstock_, trans) * int(fitted_models[media]["coef"][colnames:colnames].Coefficient)
        
        #original_spends_u = original_spends_u + X[colnames]*int(fitted_models[media]["coef"][colnames:colnames].Coefficient)
        
    response_curves_output[media] = response_curve(uplift)
    
    plt.figure(figsize=(15,5))
    plt.plot(x, response_curves_output[media], label = media)
    
    plt.scatter(media_plan[media], sales[media])
    
    point = int(round(media_plan.sum()[media:media], -4))
    plt.scatter(point, dict(zip(x,uplift))[point], 100)

    plt.xlabel("Spends (in millions)")
    plt.ylabel("Contribution to Sales")
    #plt.title(media)
    plt.title(media + ": R-square "+ str(round(fitted_models[media]["metric"]["R-Square"], 2)))
    plt.legend(loc="upper right")

# get saturation
saturation = {}
for media in media_columns:
    for adstock,row in response_curves_output.set_index("Spends").diff().iterrows():
        if row[media] < 0.01:
            saturation[media] = adstock
            break
turning_points = response_curves_output.set_index("Spends").diff().idxmax(axis = 0)
print(saturation)
print(turning_points)

x = np.arange(1000,6_900_000,1000)
#response_curves_output = pd.DataFrame(x,columns=["Spends"])

plt.figure(figsize=(15,10))
for media in media_columns:
    uplift = 0
    for colnames in fitted_models[media]["coef"].index:
        trans = colnames.split("/")[-1]
        slope = int(colnames.split("/")[0].split("slope")[-1])

        adstock_ = alpha(x, slope, slope=True)
        uplift = uplift + shape_transform(adstock_, trans) * int(fitted_models[media]["coef"][colnames:colnames].Coefficient)

    #response_curves_output[media] = response_curve(uplift)
    plt.plot(x, response_curve(uplift), label = media)
    
    # total spends from raw
    #point = int(round(media_plan_month.mean()[media:media], -4))
    #plt.scatter(point, dict(zip(x,uplift))[point], 100, cmap='viridis')
    
    # total spends from raw
    point = int(round(media_plan[:53].sum()[media:media], -4))
    plt.scatter(point, dict(zip(x,uplift))[point], 100, cmap='viridis')
    
    # turning points
    #extra_point = saturation[media]
    #extra_point = turning_points[media]
    #plt.scatter(extra_point, dict(zip(x,uplift))[extra_point], 100, marker="x",c=0)

plt.xlabel("Spends (in millions)")
plt.ylabel("Contribution to Sales")
plt.title("Response Curves: Annual Spends")
plt.legend(loc="lower right")