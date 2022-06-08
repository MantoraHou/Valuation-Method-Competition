#@------------------------------- 机器学习对多因子建模 -------------------------------------------------------------------------------------
##1.导入相关包
library(tidyr)
library(dplyr)
library(pedquant)
library(reticulate)
library(VIM)
ak <- import("akshare")

#1.1股票季度数据
stock_price <- md_stock("002568", date_range = 'max')
price <- stock_price[[1]][,c('date','close_prev')]
View(price)
price1 <- stock_price[[1]][,3:7]
library(TTR)
vol <- volatility(price1, calc = "close")
mean(vol[-1:-9,])
#---------------------------------------------------------------------------------------------------------------------------------
##2.获取股票的基本面指标
#2.1总指标
data <- ak$stock_financial_analysis_indicator("002568")
data$date <- as.Date(data$日期)
data <- data[order(data$date),]

#2.2偿债能力
changzhai_data <-md_stock_financials("002568",type = "fi1_earning")
changzhai_data <- changzhai_data[[1]] %>%
  select(date, var_name,value) %>%
  group_by(var_name) %>%
  spread(var_name, value)
View(changzhai_data)
str(changzhai_data)

#2.3盈利能力
yingli_data <-md_stock_financials("002568",type = "fi2")
yingli_data <- yingli_data[[1]] %>%
  select(date, var_name,value) %>%
  group_by(var_name) %>%
  spread(var_name, value)
View(yingli_data)
str(yingli_data)

#2.4成长能力
chengzhang_data <-md_stock_financials("002568",type = "fi3_growth")
chengzhang_data <- chengzhang_data[[1]] %>%
  select(date, var_name,value) %>%
  group_by(var_name) %>%
  spread(var_name, value)
View(chengzhang_data)
str(chengzhang_data)

#2.5营运能力
yingyun_data <- md_stock_financials("002568",type = "fi4_operation")
yingyun_data <- yingyun_data[[1]] %>%
  select(date, var_name,value) %>%
  group_by(var_name) %>%
  spread(var_name, value)
View(yingyun_data)
str(yingyun_data)

#2.6主要财务指标
caiwu_data <- md_stock_financials("002568",type = "fi0_main")
caiwu_data <- caiwu_data[[1]] %>%
  select(date, var_name,value) %>%
  group_by(var_name) %>%
  spread(var_name, value)
View(caiwu_data)
str(caiwu_data)

#---------------------------------------------------------------------------------------------------------------------------------
##3.筛选数据
#3.1流动速率:liquidity_ratio
data$liquidity_ratio <- data$流动比率
liquidity_ratio <- data %>%
  select(date,流动比率) %>%
  rename(c(liquidity_ratio=流动比率))
View(liquidity_ratio)

#3.2速动比率:quick_ratio
data$quick_ratio <- data$速动比率
quick_ratio <- data %>%
  select(date,速动比率) %>%
  rename(c(quick_ratio=速动比率))
View(quick_ratio)

#3.3资产负债率(%):ratio_of_liabilities
data$ratio_of_liabilities <- data$`资产负债率(%)`
ratio_of_liabilities <- data %>%
  select(date,ratio_of_liabilities)
View(ratio_of_liabilities)

#3.4存货周转率:rate_of_stock_turnover
data$rate_of_stock_turnover <- data$`存货周转率(次)`
rate_of_stock_turnover <- data %>%
  select(date, rate_of_stock_turnover)
View(rate_of_stock_turnover)

#3.5总资产周转率:total_assets_turnover
data$total_assets_turnover <- data$`总资产周转率(次)`
total_assets_turnover <- data %>%
  select(date, total_assets_turnover)
View(total_assets_turnover)

#3.6资产收益率:ROA
data$roa <- data$`总资产利润率(%)`
roa <- data %>%
  select(date, roa)
View(roa)

#3.7净资产收益率:ROE
data$roe <- data$`净资产报酬率(%)`
roe <- data %>%
  select(date, roe)
View(roe)

#3.8营业利润率:rate_operating_profit
data$rate_operating_profit <- data$`营业利润率(%)`
rate_operating_profit <- data %>%
  select(date, rate_operating_profit)
View(rate_operating_profit)

#3.9营业利润率:rate_of_EBIT
data$rate_of_EBIT <- data$`营业利润率(%)`
rate_of_EBIT <- data %>%
  select(date, rate_of_EBIT)
View(rate_of_EBIT)

#3.10现金比率:cash_ratio
data$cash_ratio <- data$`现金比率(%)`
cash_ratio <- data %>%
  select(date, cash_ratio)
View(cash_ratio)

#3.11净利润增长率(%):Net_profit_Growth_rate
data$Net_profit_Growth_rate <- data$`净利润增长率(%)`
Net_profit_Growth_rate <- data %>%
  select(date, Net_profit_Growth_rate)
View(Net_profit_Growth_rate)

#3.12净资产增长率(%):Growth_rate_of_net_assets
data$Growth_rate_of_net_assets <- data$`净资产增长率(%)`
Growth_rate_of_net_assets <- data %>%
  select(date, Growth_rate_of_net_assets)
View(Growth_rate_of_net_assets)

#3.13主营业务收入增长率(%): Main_business_income_growth_rate
data$Main_business_income_growth_rate <- data$`主营业务收入增长率(%)`
Main_business_income_growth_rate <- data %>%
  select(date, Main_business_income_growth_rate)
View(Main_business_income_growth_rate)

#3.14总资产增长率(%): Growth_rate_of_total_assets
data$Growth_rate_of_total_assets <- data$`总资产增长率(%)`
Growth_rate_of_total_assets <- data %>%
  select(date, Growth_rate_of_total_assets)
View(Growth_rate_of_total_assets)

#总财务数据


data_all <- data %>%
  select(date,liquidity_ratio, quick_ratio, ratio_of_liabilities,
         rate_of_stock_turnover,total_assets_turnover,roa,roe,rate_operating_profit,
         rate_of_EBIT,cash_ratio,Net_profit_Growth_rate,Growth_rate_of_net_assets,
         Main_business_income_growth_rate,Growth_rate_of_total_assets)
data_all$date <- as.Date(data_all$date) 
data_all <- left_join(data_all, price, by = 'date')
View(data_all)
write.csv(data_all, "C:\\Users\\13407\\Desktop\\基金管理案例大赛\\data.csv")
write.csv(price, "C:\\Users\\13407\\Desktop\\基金管理案例大赛\\stock.csv")
data_all <- read.csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\data.csv")
data_all <- data_all[,c(-1,-16)] 
View(data_all)
aggr(data_all)
#--------------------------------------------------------------------------------------------------------------------------------
##4.模型检验
formula <- as.formula(data_all[,c(-1)])
for (i in names(data_all[,-1])) {
  data_all[,i] <- data_all[,i]/100
}
View(data_all)

data1 <- as.data.frame(scale(data_all[,c(-1)]))
model <- lm(formula, data = data_all)
summary(model)

#---------------------------------------------------------------------------------
str(data)
data <- data[c(-1:-9),c(-1,-5,-11,-22,-26:-27,-31,-40,-50,-52:-53,-55:-57,-60:-61,-66,-69:-86)]
write.csv(data, "C:\\Users\\13407\\Desktop\\基金管理案例大赛\\data1.csv")
data <- read.csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\data1.csv")
data$date <- as.Date(data$date)
#data <- left_join(data, price, by = "date")

#----------------------------------------------------------------------------------------
#模型训练和预测
formula <- as.formula(data[,-1])
model <- lm(formula, data)
summary(model)

train <- data[c(1:30),]
test <- data[c(31:40),]
random <- randomForest(formula, data[c(-41:-42),])
pred <- predict(random, data)
res <- cbind(data[,1:2], pred)
plot(res$date,res$close_prev, main = "The Accuary and The Prediction", xlab = "Date", ylab = "Value")
lines(res$date,res$pred, col = "blue",lwd = 2)
lines(res$date,res$close_prev, col="red", lwd=2)
res$gap <- res$close_prev - res$pred
mean(res$gap);sd(res$gap)
res$rnorm <-rnorm(41,-0.3943476, 9.627946);res$pred9 <- res$pred + res$rnorm;lines(res$date,res$pred9)
mtext("The Accuary: red \n The Prediction: black",side = 1,adj = 1, line = -1.5,col = "orange", font = 2)

b <-as.data.frame(rep(82.70, 10000))
colnames(b) <- c("mean")
b$rnorm <- rnorm(10000, -0.3943476, 9.627946)
b$c <- b$mean+b$rnorm
plot(b$c, xlab = "Frequency", ylab = "Stock Prices")
b$d <- ifelse((b$c <= 82.70),b$c, 0)
e <- b[b$d != 0,]
mean(e$d)


sseVol = aggregate(data$close_prev, as.numeric(format(index(data$close_prev), "%Y")),
                   function(ss) coredata(tail(TTR:::volatility(
                     ss,
                     n=NROW(ss),
                     calc="close"), 1)))
#----------------------------------------------------------------------------
#期权价格求取
bsvalue = function(S,d1, d2, X){
  c = S*pnorm(d1) - X*pnorm(d2)
  return(c)
}
bsvalue(74.84, 0.5096,-0.620665,34.32)
bsvalue(82.70, 0.5096,-0.620665,26.73)
#参数
Rvalue = function(a,d1,d2,b){
  r = a*pnorm(d1) + b*pnorm(d2)
  return(r)
}
Rvalue(4.4904,0.5096,-0.620665,1.14)
Rvalue(4.962,0.5096,-0.620665,0.64)












