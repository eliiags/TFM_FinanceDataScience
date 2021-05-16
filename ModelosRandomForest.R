# ************************************
# TFM - EAE Business School
#     - Master Finance & Data Science
# Autoras:
#   - Elianni Aguero
#   - Amaia Peñagaricano
#   - Marta Martin
#   - Celia Montoya
#   - Irene La Torre
# 
# Modelos de Arboles - Random Forest
# ************************************

library(readxl)
library(dummies)
library(lattice)
library(ggplot2)
library(caret)
library(e1071)
library(foreach)
library(iterators)
library(parallel)
library(doParallel)
library(nnet)
library(MASS)
library(reshape)
library(dplyr)
library(pROC)


set.seed(12345)
control <- trainControl(method="LGOCV", p=0.7, number=1, savePredictions="all", classProbs=TRUE)
rfgrid  <- expand.grid(mtry=c(3, 6, 9, 12, 15, 18, 21, 24, 28, 32, 43))

clust <- parallel::makeCluster(5)
doParallel::registerDoParallel(cl = clust)
rf1<-train(y~., data=bank,
          method="rf",
          ntree=200,
          sampsize=2000,
          nodesize=100,
          replace=TRUE, 
          importance=TRUE,
          trControl=control,
          tuneGrid=rfgrid)
foreach::registerDoSEQ()
parallel::stopCluster(clust)

plot(rf1$pred)

# Importancia de las variable
tabla1 <- as.data.frame(rf1$finalModel$importance)
tabla1 <- tabla1[order(-tabla1$MeanDecreaseAccuracy),]
tabla1

par(cex=0.5, las=2)
par(mar=c(15, 7, 4.1, 2.1))#
barplot(tabla1$MeanDecreaseAccuracy, names.arg=row.names(tabla1), main="Importancia de las variables", cex.names=1.12)


# validacion cruzada 
rf1 <- cruzadarfbin(data=bank, vardep="y",
                    listconti=c("age", "job.admin", "job.blue_collar", "job.entrepreneur", 
                                "job.management", "job.retired", "job.self_employed", "job.services", 
                                "job.technician", "job.unemployed", "marital.divorced", "marital.married", 
                                "marital.single", "marital.unknown", "education.basic.4y", "education.basic.6y", 
                                "education.basic.9y", "education.high.school", "education.professional.course", 
                                "education.university.degree", "education.unknown", "default.no", 
                                "default.unknown", "default.yes", "contact.cellular", "contact.telephone", 
                                "month.apr", "month.aug", "month.jul", "month.jun", "month.may", 
                                "month.nov", "month.other", "campaign", "poutcome.failure", "poutcome.nonexistent", 
                                "poutcome.success", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                                "euribor3m", "nr.employed"),
                    listclass=c(""),
                    grupos=4, sinicio=1234, repe=5, nodesize=10,
                    mtry=3, ntree=200, replace=TRUE, sampsize=2000)
rf1$modelo="rf_3"

rf2 <- cruzadarfbin(data=bank, vardep="y",
                    listconti=c("age", "job.admin", "job.blue_collar", "job.entrepreneur", 
                                "job.management", "job.retired", "job.self_employed", "job.services", 
                                "job.technician", "job.unemployed", "marital.divorced", "marital.married", 
                                "marital.single", "marital.unknown", "education.basic.4y", "education.basic.6y", 
                                "education.basic.9y", "education.high.school", "education.professional.course", 
                                "education.university.degree", "education.unknown", "default.no", 
                                "default.unknown", "default.yes", "contact.cellular", "contact.telephone", 
                                "month.apr", "month.aug", "month.jul", "month.jun", "month.may", 
                                "month.nov", "month.other", "campaign", "poutcome.failure", "poutcome.nonexistent", 
                                "poutcome.success", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                                "euribor3m", "nr.employed"),
                    listclass=c(""),
                    grupos=4, sinicio=1234, repe=5, nodesize=10,
                    mtry=6, ntree=200, replace=TRUE, sampsize=2000)
rf2$modelo="rf_6"

rf3 <- cruzadarfbin(data=bank, vardep="y",
                    listconti=c("age", "job.admin", "job.blue_collar", "job.entrepreneur", 
                                "job.management", "job.retired", "job.self_employed", "job.services", 
                                "job.technician", "job.unemployed", "marital.divorced", "marital.married", 
                                "marital.single", "marital.unknown", "education.basic.4y", "education.basic.6y", 
                                "education.basic.9y", "education.high.school", "education.professional.course", 
                                "education.university.degree", "education.unknown", "default.no", 
                                "default.unknown", "default.yes", "contact.cellular", "contact.telephone", 
                                "month.apr", "month.aug", "month.jul", "month.jun", "month.may", 
                                "month.nov", "month.other", "campaign", "poutcome.failure", "poutcome.nonexistent", 
                                "poutcome.success", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                                "euribor3m", "nr.employed"),
                    listclass=c(""),
                    grupos=4, sinicio=1234, repe=5, nodesize=10,
                    mtry=9, ntree=200, replace=TRUE, sampsize=2000)
rf3$modelo="rf_9"

rf4 <- cruzadarfbin(data=bank, vardep="y",
                    listconti=c("age", "job.admin", "job.blue_collar", "job.entrepreneur", 
                                "job.management", "job.retired", "job.self_employed", "job.services", 
                                "job.technician", "job.unemployed", "marital.divorced", "marital.married", 
                                "marital.single", "marital.unknown", "education.basic.4y", "education.basic.6y", 
                                "education.basic.9y", "education.high.school", "education.professional.course", 
                                "education.university.degree", "education.unknown", "default.no", 
                                "default.unknown", "default.yes", "contact.cellular", "contact.telephone", 
                                "month.apr", "month.aug", "month.jul", "month.jun", "month.may", 
                                "month.nov", "month.other", "campaign", "poutcome.failure", "poutcome.nonexistent", 
                                "poutcome.success", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                                "euribor3m", "nr.employed"),
                    listclass=c(""),
                    grupos=4, sinicio=1234, repe=5, nodesize=10,
                    mtry=12, ntree=200, replace=TRUE, sampsize=2000)
rf4$modelo="rf_12"

rf5 <- cruzadarfbin(data=bank, vardep="y",
                    listconti=c("age", "job.admin", "job.blue_collar", "job.entrepreneur", 
                                "job.management", "job.retired", "job.self_employed", "job.services", 
                                "job.technician", "job.unemployed", "marital.divorced", "marital.married", 
                                "marital.single", "marital.unknown", "education.basic.4y", "education.basic.6y", 
                                "education.basic.9y", "education.high.school", "education.professional.course", 
                                "education.university.degree", "education.unknown", "default.no", 
                                "default.unknown", "default.yes", "contact.cellular", "contact.telephone", 
                                "month.apr", "month.aug", "month.jul", "month.jun", "month.may", 
                                "month.nov", "month.other", "campaign", "poutcome.failure", "poutcome.nonexistent", 
                                "poutcome.success", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                                "euribor3m", "nr.employed"),
                    listclass=c(""),
                    grupos=4, sinicio=1234, repe=5, nodesize=10,
                    mtry=21, ntree=200, replace=TRUE, sampsize=2000)
rf5$modelo="rf_21"



union <- rbind(rf1, rf2, rf3, rf4, rf5)
par(cex.axis=1.2)
par(mar=c(5, 7, 4.1, 2.1))#
boxplot(data=union, tasa~modelo, main="Tasa de Fallos FR")
boxplot(data=union, auc~modelo,  main="Curva ROC FR", col="lightgrey")



# mejor modelo
set.seed(12345)
control <- trainControl(method="repeatedcv", number=4, repeats=5, savePredictions="all", classProbs=TRUE)
rfgrid  <- expand.grid(mtry = c(12))

clust <- parallel::makeCluster(5)
doParallel::registerDoParallel(cl = clust)
rf<-train(y~., data=bank,
          method="rf",
          ntree=200,
          sampsize=2000,
          nodesize=100,
          replace=TRUE, 
          importance=TRUE,
          trControl=control,
          tuneGrid=rfgrid)
foreach::registerDoSEQ()
parallel::stopCluster(clust)


# Curva Lift
library(data.table)
curva_lift = data.table(compra=rf$pred$obs, score=rf$pred$pred)
ordenacion = curva_lift[order(-curva_lift$score),]
(total_true=dim(ordenacion[ordenacion$compra=="Yes",])[1])

lift_table=data.frame(decil=NULL, lift=NULL)
for(i in c(1:10)){
  porcentaje=0.1*i
  data=head(ordenacion, as.integer(dim(ordenacion)[1]*porcentaje))
  (total_d=length(data[data$compra=="Yes",]$score))
  (Lift=total_d/(total_true*porcentaje))
  lift_table=rbind(lift_table, c(porcentaje*10, Lift))
}

colnames(lift_table)=c("decil", "Lift")
lift_table

plot(Lift~decil, lift_table, col="red",
     main=expression("Lift vs. decil"),
     xlab=expression("Decil"),
     ylab=expression("Lift"))







# ********************************
# Validacion cruzada random forest
# ********************************

cruzadarfbin <- function(data=data,vardep="vardep",
                         listconti="listconti",
                         listclass="listclass",
                         grupos=4, sinicio=23461, repe=10, nodesize=20,
                         mtry=2, ntree=50, replace=TRUE, sampsize=400) { 
  
  library(dummies)
  library(MASS)
  library(reshape)
  library(caret)
  library(dplyr)
  library(pROC)  
  
  if  (listclass!=c("")) {
    databis <- data[, c(vardep, listconti, listclass)]
    databis <- dummy.data.frame(databis, listclass, sep=".")
  }  
  else   {  
    databis <- data[, c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  means <-apply(data[,listconti],2,mean)
  sds<-sapply(data[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  datacon<-scale(data[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(data)%in%listconti)
  databis<-cbind(datacon,data[,-numerocont,drop=FALSE ])
  
  databis[, vardep] <- as.factor(databis[, vardep])
  
  formu <- formula(paste("factor(", vardep, ")~.", sep=""))
  
  # Preparo caret   
  set.seed(sinicio)
  control <- trainControl(method="repeatedcv", number=grupos, repeats=repe,
                          savePredictions="all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  rfgrid <-expand.grid(mtry=mtry)
  
  clust <- parallel::makeCluster(6)
  doParallel::registerDoParallel(cl = clust)
  rf<- train(formu, data=databis,
             method="rf", trControl=control,
             tuneGrid=rfgrid, nodesize=nodesize, replace=replace,
             ntree=ntree, sampsize=sampsize)
  foreach::registerDoSEQ()
  parallel::stopCluster(clust)
  
  # Resultados
  preditest<-rf$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold  <- sapply(preditest$prueba, "[", 1)
  preditest$Rep   <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  
  tasafallos<-function(x,y) {
    confu<-confusionMatrix(x,y)
    tasa<-confu[[3]][1]
    return(tasa)
  }
  
  # Aplicamos funciÃ³n sobre cada RepeticiÃ³n
  
  t_forest<-preditest %>%
    group_by(Rep) %>%
    summarize(tasa=1-tasafallos(pred,obs))
  
  # CalculamoS AUC  por cada RepeticiÃ³n de cv 
  # Definimnos funciÃ³n
  
  auc<-function(x,y) {
    curvaroc<-roc(response=x,predictor=y)
    auc<-curvaroc$auc
    return(auc)
  }
  
  # Aplicamos funciÃ³n sobre cada RepeticiÃ³n
  
  forestbis<-preditest %>%
    group_by(Rep) %>%
    summarize(auc=1*auc(preditest$obs,preditest$Yes))
  
  # Unimos la info de auc y de tasafallos
  
  t_forest$auc<-forestbis$auc
  
  return(t_forest)
  
}
