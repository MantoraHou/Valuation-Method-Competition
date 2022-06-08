#@-------------------------------------------- 各财务指标预测和画图 ------------------------------------------------------------------
library(reticulate)
library(pedquant)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tidyr)
library(xts)
ak <- import("akshare")

#1.获取公司股价和沪深300走势图
#Stock_p <- md_stock(c("002568","^000300"), date_range = "3y");stock <- Stock_p[[1]];Index <- Stock_p[[2]];rm(Stock_p)
#Stock_p <- md_stock(c("002568","^000300"), date_range = "3y")
#Stock_p[[1]]$close_prev <- xts(Stock_p[[1]]$close_prev, order.by = as.Date(Stock_p[[1]]$date))
#Stock_p[[2]]$close_prev <- xts(Stock_p[[2]]$close_prev, order.by = as.Date(Stock_p[[2]]$date))
#val1 <- (diff(Stock_p[[1]]$close_prev, lag =1)/lag(Stock_p[[1]]$close_prev,1))*100;val1 <- val1[-1,]
#val2 <- (diff(Stock_p[[2]]$close_prev, lag =1)/lag(Stock_p[[2]]$close_prev,1))*100;val2 <- val1[-1,]
Stock_p <- md_stock(c("002568","^000300"), date_range = "3m")
pq_plot(Stock_p[[1]], chart_type = 'line', addti = list(sma = list(n = 200),sma = list(n = 50),macd = list()))
pq_plot(Stock_p, x='close_prev', multi_series = list(nrow=1, ncol=1), cumreturns=TRUE) #累计收益

#2.公司上市以来状况
data1 <- ak$stock_financial_analysis_indicator("002568")
data1$date <- as.Date(data1$日期);data1 <- data1[order(data1$date),]; data1 <- data1[c(-1:-5),]
data2 <- (md_stock_financials("002568", type = "fs2_balance"))[[1]] %>%select(date, var_name,value) %>% group_by(var_name) %>% spread(var_name, value); data2 <- data2[c(-1:-2),]
data3 <- (md_stock_financials("002568", type = "fs3_cashflow"))[[1]] %>%select(date, var_name,value) %>% group_by(var_name) %>% spread(var_name, value); data3 <- data3[c(-1:-2),]
data4 <- (md_stock_financials("002568", type = "fs1_income"))[[1]] %>%select(date, var_name,value) %>% group_by(var_name) %>% spread(var_name, value); data4 <- data4[c(-1:-2),]
data5 <- (md_stock_financials("002568", type = "fi0_main"))[[1]] %>%select(date, var_name,value) %>% group_by(var_name) %>% spread(var_name, value); data5 <- data5[c(-1:-2),]
data6 <- (md_stock_financials("002568", type = "fs0_summary"))[[1]] %>%select(date, var_name,value) %>% group_by(var_name) %>% spread(var_name, value); data6 <- data6[c(-1:-2),]
data7 <- (md_stock_financials("002568", type = "fi1_earning"))[[1]] %>%select(date, var_name,value) %>% group_by(var_name) %>% spread(var_name, value); data7 <- data7[c(-1:-2),]


#2.1营业收入和主营业务收入及增速情况
revenue <- xts(data2$`营业收入(万元)`, order.by = as.Date(data2$date))
colnames(revenue) <- c("revenue");revenue$label <- as.factor(quarter(revenue));revenue$label_y <- year(revenue)
revenue <- revenue[revenue$label == '4' | revenue$label_y == '2021',]; revenue<-revenue[revenue$label == '4' | revenue$label == '3',]
#revenue$label_y <- ifelse((revenue$label_y == "2021"),"2021Q3", revenue$label_y)
revenue <- as.data.frame(revenue);revenue$label_y <- factor(revenue$label_y, labels = c("2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021Q3"))
revenue$grow <- ((revenue$revenue/lag(revenue$revenue,1))-1)*100;revenue$grow <- ifelse((is.na(revenue$grow) == TRUE), 0, revenue$grow)
revenue$growth <- factor(revenue$grow, labels = c("0%", "8.31%","39.24%","-21.77%","22.36%","1395.86%","-60.64%",
                                                "26.63%","4.95%","19.38%", "31.20%","-0.62%"))
ggplot(data = revenue, aes(x = label_y)) +
  geom_bar(aes(y =revenue),stat = "identity",fill = "steelblue", color ="blue") +
  geom_line(aes(y = grow*150, group = 1,color = "purple")) + theme_classic() +labs(x="", y="") +
  scale_y_continuous(sec.axis = sec_axis(~.*0.006, breaks = c(seq(-200,1500,200))))

#2.2 近年来公司利润、现金流以及利润，收入的增速对比
return <- xts(data2$`净利润(万元)`,order.by = as.Date(data2$date))
colnames(return) <- c("return");return$label <- as.factor(quarter(return));return$label_y <- year(return)
return <- return[return$label == '4' | return$label_y == '2021',]; return<-return[return$label == '4' | return$label == '3',]
return <- as.data.frame(return);return$label_y <- factor(return$label_y, labels = c("2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021Q3"))
cash_flow <-  xts(data3$`经营活动现金流入小计(万元)`- data3$`经营活动现金流出小计(万元)`,order.by = as.Date(data2$date)) 
colnames(cash_flow) <- c("cash_flow");cash_flow$label <- as.factor(quarter(cash_flow));cash_flow$label_y <- year(cash_flow)
cash_flow <- cash_flow[cash_flow$label == '4' | cash_flow$label_y == '2021',]; cash_flow<-cash_flow[cash_flow$label == '4' | cash_flow$label == '3',]
cash_flow <- as.data.frame(cash_flow);cash_flow$label_y <- factor(cash_flow$label_y, labels = c("2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021Q3"))
data_re_cf <- left_join(return, cash_flow, by = "label_y") %>%
  gather(key = "kind", value = "measure", return, cash_flow)
data_re_cf$measure <- data_re_cf$measure/100

ggplot(data_re_cf, aes(x=label_y, y=measure)) + 
  geom_col(aes(fill=kind), position="dodge") + 
  geom_text(aes(label=measure, y=measure+10),position=position_dodge(0.1), vjust=0,hjust = 0.1) +
  theme_classic() +labs(x="", y="")

#2.3毛利率和净利率
gpr <- xts(data1$`销售毛利率(%)`,order.by = as.Date(data1$date))
ngpr <- xts(data1$`销售净利率(%)`, order.by = as.Date(data1$date))
write.csv(gpr, "C:\\Users\\13407\\Desktop\\基金管理案例大赛\\gpr.csv")
write.csv(ngpr, "C:\\Users\\13407\\Desktop\\基金管理案例大赛\\ngpr.csv")
gpr <- read.csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\gpr.csv") %>%
  xts(order.by = as.Date(data1$date))
gpr <- gpr[,c(-1)]; colnames(gpr) <- c("gpr")
data_gpr_ngpr <- cbind(gpr, ngpr)
par(mfrow = c(1,1))
plot.xts(data_gpr_ngpr, main = "毛利率和净利率变化图（%）\n 毛利率：黑色 \n 净利率：红色")

#2.4 主营业务利润率(%)、主营利润比重、主营业务收入增长率
Main_return <-xts(data1$`主营业务利润率(%)`,order.by = as.Date(data1$date))
Main_weight <-xts(data1$主营利润比重,order.by = as.Date(data1$date))
Main_return_grow <-xts(data1$`主营业务收入增长率(%)`,order.by = as.Date(data1$date))
data_main <- cbind(Main_return,Main_return_grow,Main_weight)[c(-1:-4),]
write.csv(data_main, "C:\\Users\\13407\\Desktop\\基金管理案例大赛\\data_main.csv")
data_main <- read.csv("C:\\Users\\13407\\Desktop\\基金管理案例大赛\\data_main.csv")
data_main <- xts(data_main, order.by = as.Date(data1[c(-1:-4),]$date))[,c(-1)]

plot(data_main[,c(1)],main = "主营业务利润率(%)")
plot(data_main[,c(2)],main = "主营业务收入增长率(%)")
plot(data_main[,c(3)], main = "主营利润比重")
plot(data_main[,c(2:3)], main = "主营利润比重:red \n 主营业务收入增长率(%): black")


