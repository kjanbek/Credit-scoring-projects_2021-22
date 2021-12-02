library(quantmod)
library(tidyverse)
library(tidyquant)
library(dplyr)
library(readxl)
library(ggplot2)
library(aTSA)
library(forecast)
library(tsDyn)
library(xts)
library(stats)
library(graphics)
library(naniar)
library(deal)
library(FactoMineR)
library(factoextra)
library(ggplot2)
library(party)
library(rpart)
library(randomForest)

#Import data
Defaut2000 <- read.delim("C:/Users/khali/Desktop/Credit scoring - Project/defaut2000_foruploadinr.txt")

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
Defaut2000_cleaned <- replace_with_na_all(data = Defaut2000_Sorted, condition = ~.x == -99.99)

#Data frame 1: estimation (training) sample
Estimation_Defaut2000 <- Defaut2000_cleaned[seq_len(nrow(Defaut2000_cleaned)) %% 2 == 1, ] 

#Data frame 2: validation sample
Validation_Defaut2000 <- Defaut2000_cleaned[seq_len(nrow(Defaut2000_cleaned)) %% 2 == 0, ] 

#Principal Component Analysis (07/11)

Estimation_Defaut2000[is.na(Estimation_Defaut2000)] <- 0 #PCA doesn't allow missing values

PCA <- prcomp(Estimation_Defaut2000[,-1], scale. = TRUE)
summary(PCA)

plot(PCA, type="l")
plot(PCA)

biplot(PCA, scale=0)

Estimation_DefautData2 <- cbind(Estimation_Defaut2000,PCA$x[,1:2])
head(Estimation_DefautData2)

ggplot(Estimation_DefautData2, aes(PC1, PC2, col = yd, fill = yd))+
  stat_ellipse(geom="polygon", col="black", alpha=0.5) +
  geom_point(shape = 21, col = "black")

cor(Estimation_Defaut2000[,-1], Estimation_DefautData2[,16:17])

#Validation sample PCA
Validation_Defaut2000[is.na(Validation_Defaut2000)] <- 0 #PCA doesn't allow missing values

PCA_V <- prcomp(Validation_Defaut2000[,-1], scale. = TRUE)
summary(PCA_V)

plot(PCA_V, type="l")
plot(PCA)

biplot(PCA_V, scale=0)

Validation_DefautData2 <- cbind(Validation_Defaut2000,PCA$x[,1:2])
head(Estimation_DefautData2)

ggplot(Validation_DefautData2, aes(PC1, PC2, col = yd, fill = yd))+
  stat_ellipse(geom="polygon", col="black", alpha=0.5) +
  geom_point(shape = 21, col = "black")

cor(Estimation_Defaut2000[,-1], Estimation_DefautData2[,16:17])

#Total dataset PCA

Defaut2000[is.na(Defaut2000)] <- 0 #PCA doesn't allow missing values

PCA_T <- prcomp(Defaut2000[,-1], scale. = TRUE)
PCA_T
summary(PCA_T)

plot(PCA_T, type="l")
plot(PCA)

biplot(PCA_T, scale=0)

Total_DefautData <- cbind(Defaut2000,PCA_T$x[,1:2])
head(Total_DefautData)

ggplot(Total_DefautData, aes(PC1, PC2, col = yd, fill = yd))+
  stat_ellipse(geom="polygon", col="black", alpha=0.5) +
  geom_point(shape = 21, col = "black")

cor(Defaut2000[,-1], Total_DefautData[,16:17])

#Cluster analysis (KNN)

plot(Defaut2000)

#......Normalizing data........
Defaut2000_Scaled <- scale(Defaut2000[,-1])

#......Running K-means clustering.......
FitK <- kmeans(Defaut2000_Scaled, 3) #We have 2 categories in our data (yd ==0 and 1), so we start off with K=2
FitK

plot(Defaut2000,col=FitK$cluster)

#......Choosing the right number Of K.....

k <- list()
for(i in 1:10){
  k[[i]] <- kmeans(Defaut2000_Scaled,i)
}

k

betweenss_totss <- list()
for(i in 1:10){
  betweenss_totss[[i]] <- k[[i]]$betweenss/k[[i]]$totss
}

plot(1:10, betweenss_totss, type = "b",
     ylab = "Between SS / Total SS", xlab = "Clusters (k)")
