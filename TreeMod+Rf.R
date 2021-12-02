library(tidyverse)
library(tidyr)
library(haven)
library(naniar)
library(ROCR)
library(rpart)
library(rpart.plot)
library(dplyr)
library(deal)
library(bnlearn)
library(party)
library(randomForest)
library(DescTools)
library(aod)
library(DAAG)
library(jtools)
library(huxtable)
library(officer)
library(flextable)
library(randomForestExplainer)


Defaut2000 <- read_dta("defaut2000bis.dta")

#Define variables
yd <- Defaut2000[,1]
tdta <- Defaut2000[,2]
reta <- Defaut2000[,3]
opita <- Defaut2000[,4]
ebita <- Defaut2000[,5]
lsls <- Defaut2000[,6]
lta <- Defaut2000[,7]
gempl <- Defaut2000[,8]
invsls <- Defaut2000[,9]
nwcta <- Defaut2000[,10]
cacl <- Defaut2000[,11]
qacl <- Defaut2000[,12]
fata <- Defaut2000[,13]
ltdta <- Defaut2000[,14]
mveltd <- Defaut2000[,15]

#Sort data
Defaut2000_Sorted <- Defaut2000[order(yd, ebita),]
head(Defaut2000_Sorted)

#Remove outliers
Defaut2000_cleaned <- replace_with_na_all(data = Defaut2000_Sorted, condition = ~.x < -99)
Defaut2000_cleaned[is.na(Defaut2000_cleaned)] <- 0

#Data frame 1: estimation (training) sample
Estim <- Defaut2000_cleaned[seq_len(nrow(Defaut2000_cleaned)) %% 2 == 1, ] 

#Data frame 2: validation sample
Test <- Defaut2000_cleaned[seq_len(nrow(Defaut2000_cleaned)) %% 2 == 0, ] 

#First logit model
Model1 <- glm(yd~., data = Estim, family = binomial(link = "logit"))
summary(Model1)
export_summs(Model1, Robust= "HC1", digits = 5, to.file = "docx", file.name = "test.docx")
summ(Model1, confint = TRUE, digits = 3)

confint.default(Model1)
exp(cbind(OR = coef(Model1), confint.default(Model1)))

logLik(Model1)
PseudoR2(Model1)

Score1 <- predict(Model1, type='response',Test)
Pred1 <- prediction(Score1, Test$yd)
Perf1 <- performance(Pred1, "tpr","fpr")
plot(Perf1, main = "Logit Model")


#Recursive Partitioning tree
#Classification tree
Estim$yd <- ifelse(Estim$yd == 1, "Default", "No Default")
Test$yd <- ifelse(Test$yd == 1, "Default", "No Default")
Fit1 <- rpart(yd~., data = Estim, method = "class")
rpart.plot(Fit1, type = 4, clip.right.labs= FALSE, branch=.3, tweak = 1.2)
rpart.rules(Fit1)
printcp(Fit1)
plotcp(Fit1)

prunefit1 <- prune(Fit1, cp= Fit1$cptable[which.min(Fit1$cptable[,"xerror"]),"CP"])
rpart.rules(prunefit1)
rpart.plot(prunefit1, type = 4, clip.right.labs= FALSE, branch=.3, tweak = 1.2)


Score2 <- predict(Fit1, type='vector',Estim)
Pred2 <- prediction(Score2[1:90], Test$yd)
Perf2 <- performance(Pred2, "tpr", "fpr")
Score2bis <- predict(prunefit1, type='vector',Estim)
Pred2bis <- prediction(Score2bis[1:90], Test$yd)
Perf2bis <- performance(Pred2bis, "tpr", "fpr")


plot(Perf2, col = "red", lty=1);
plot(Perf2bis, col = "blue", lty =2, add = TRUE);
legend(0.6,0.6,c("Default tree", "Pruned tree"), col = c("red", "blue"), lwd =2)

auc2 <- performance(Pred2, 'auc')@y.values[[1]]
auc2

auc2bis <- performance(Pred2bis, 'auc')@y.values[[1]]
auc2bis

#regression tree 
Estim <- Defaut2000_cleaned[seq_len(nrow(Defaut2000_cleaned)) %% 2 == 1, ] 
Test <- Defaut2000_cleaned[seq_len(nrow(Defaut2000_cleaned)) %% 2 == 0, ] 

Fit2 <- rpart(yd~., data = Estim, method = "anova")
rpart.plot(Fit2, type = 4, clip.right.labs= FALSE, branch=.3, tweak = 1.2)
rpart.rules(Fit2)
printcp(Fit2)
plotcp(Fit2)

prunefit2 <- prune(Fit2, cp= Fit2$cptable[which.min(Fit2$cptable[,"xerror"]),"CP"])
rpart.rules(prunefit2)
rpart.plot(prunefit2, type = 4, clip.right.labs= FALSE, branch=.3, tweak = 1.2)


Score3 <- predict(Fit2, type='vector',Estim)
Pred3 <- prediction(Score3[1:90], Test$yd)
Perf3 <- performance(Pred3, "tpr", "fpr")
Score3bis <- predict(prunefit2, type='vector',Estim)
Pred3bis <- prediction(Score3bis[1:90], Test$yd)
Perf3bis <- performance(Pred3bis, "tpr", "fpr")



plot(Perf3, col = "red", lty=1);
plot(Perf3bis, col = "blue", lty =2, add = TRUE);
legend(0.6,0.6,c("Default tree", "Pruned tree"), col = c("red", "blue"), lwd =2)

auc3 <- performance(Pred3, 'auc')@y.values[[1]]
auc3

auc3bis <- performance(Pred3bis, 'auc')@y.values[[1]]
auc3bis



#conditional inference trees

cfit1 <- ctree(yd~., data= Estim)
plot(cfit1, terminal_panel = node_barplot(cfit1))
print(cfit1)
cfit1[1]

result <- as.data.frame(do.call("rbind", treeresponse(cfit1, newdata = Test)))
result
Score4 <- result$V1
Pred4 <- prediction(Score4, Test$yd)
Perf4 <- performance(Pred4, "tpr", "fpr")
plot(Perf4)

auc4 <- performance(Pred4, 'auc')@y.values[[1]]
auc4


#random forest

Defaut2000_cleaned$yd <- as.numeric(Defaut2000_cleaned$yd)
set.seed(42)
TestRF <- randomForest(yd~., data= Defaut2000_cleaned, importance = TRUE, proximity = TRUE, ntree=500, keep.forest= TRUE)
print(TestRF)
which.min(TestRF$mse)
plot(TestRF)


set.seed(42)
RandomForest <- randomForest(yd~., data= Defaut2000_cleaned, importance = TRUE, proximity = TRUE, ntree=458, keep.forest= TRUE)
print(RandomForest)
which.min(RandomForest$mse)
plot(RandomForest)
varImpPlot(RandomForest)


min_depth_frame <- min_depth_distribution(RandomForest)
min_depth_frame
plot_min_depth_distribution(min_depth_frame)

plot_multi_way_importance(RandomForest, size_measure = "no_of_nodes")
plot_multi_way_importance(RandomForest, x_measure = "mse_increase", y_measure = "node_purity_increase", size_measure = "p_value")

plot_importance_ggpairs(RandomForest)
plot_importance_rankings(RandomForest)

vars <- important_variables(RandomForest, k = 5, measures = c("mean_min_depth", "no_of_trees"))
vars
interactions_frame <- min_depth_interactions(RandomForest, vars)
plot_min_depth_interactions(interactions_frame)

interactions_frame2 <- min_depth_interactions(RandomForest, vars, mean_sample = "relevant_trees", uncond_mean_sample = "relevant_trees")
plot_min_depth_interactions(interactions_frame2)

Estim2 <- as.data.frame(Estim)
plot_predict_interaction(RandomForest, Estim2, c("opita"), c("ebita"))


ScoreRF <- predict(RandomForest, Defaut2000_cleaned, type = 'response')
PredRF <- prediction(ScoreRF, as.numeric(Defaut2000_cleaned$yd))
PerfRF <- performance(PredRF, "tpr", "fpr")
plot(PerfRF, main = "Random forest model")

explain_forest(RandomForest, interactions = TRUE, data = Defaut2000_cleaned)

#improving logit using random forest 

Model2 <- glm(yd~. + opita:ebita + opita:lta + opita:opita + opita:qacl, data=Estim, family= binomial())
summary(Model2)
summ(Model2)
export_summs(Model2, Robust= "HC1", digits = 5, to.file = "docx", file.name = "test.docx")

Score5 <- predict(Model2, type ='response', Estim)
Pred5 <- prediction(Score5[1:90], Test$yd)
Perf5 <- performance(Pred5, "tpr", "fpr")
plot(Perf5, main ="Improved logit model")

#Graph with all the ROC curves

plot(Perf1, col='red', lty=1, main='ROC Curves');
plot(Perf2, col='chartreuse', lty=2, add=TRUE);
plot(Perf3, col='black', lty=3, add=TRUE);
plot(Perf4, col='orange', lty=4, add=TRUE);
plot(Perf5, col='blue', lty=5, add=TRUE);
legend(0.6,0.6,c('Simple logit', 'Classification tree','Regression tree' ,'Conditional Inference tree', 'Logit with interactions'), col=c('red', 'chartreuse', 'black', 'orange', 'blue'), lwd =2)

#computing the area under the ROC curve
auc1 <- performance(Pred1, "auc")
auc1 <- auc1@y.values[[1]]
auc1

auc2 <- performance(Pred2, "auc")
auc2 <- auc2@y.values[[1]]

auc3 <- performance(Pred3, "auc")
auc3 <- auc3@y.values[[1]]

auc4 <- performance(Pred4, "auc")
auc4 <- auc4@y.values[[1]]
    
auc5 <- performance(Pred5, "auc")
auc5 <- auc5@y.values[[1]]
auc5
  
cbind(auc1, auc2, auc3, auc4, auc5)

#Cross validation for logit models

H1 <-CVbinary(obj=Model1, rand=NULL, nfolds=100,print.details=TRUE)

H2 <-CVbinary(obj=Model2, rand=NULL, nfolds=100,print.details=TRUE)

#Bayesian Networks (max-min hill climbing)

bn_df <- data.frame(Estim)
res <- hc(bn_df)
plot(res)
bn.mod <- bn.fit(res, data = Estim)
bn.mod

Model3 <- lm(yd~., data = Estim, na.action = na.omit)
summary(Model3)
Model4 <- lm(yd~tdta, data = Estim, na.action = na.omit)
summary(Model4)
