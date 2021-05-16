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
# Modelos de Redes Neuronales 
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

# Cargamos los datos
bank <- as.data.frame(read_excel("bank-additional-full.xlsx"))

continuas  <-c("age", "campaign", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
categoricas<-c("job", "marital", "education", "default", "contact", "month", "poutcome")

bank$job       <- as.factor(bank$job)
bank$marital   <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$default   <- as.factor(bank$default)
bank$contact   <- as.factor(bank$contact)
bank$month     <- as.factor(bank$month)
bank$poutcome  <- as.factor(bank$poutcome)
bank$y         <- as.factor(bank$y)

bank$euribor3m     <- as.integer(bank$euribor3m)
bank$emp.var.rate  <- as.integer(bank$emp.var.rate)
bank$cons.conf.idx <- as.integer(bank$cons.conf.idx)

summary(bank)

# Pasar las categóricas a dummies
bank <- dummy.data.frame(bank, categoricas, sep=".")
# Se debe definir los valores de salida con valores alfanumericos (yes, no)
bank$y <- ifelse(bank$y==1, "Yes", "No")


set.seed(12345)
# training-test
control<-trainControl(method="LOOCV", p=0.7, savePredictions = "all", classProbs=TRUE) 

avnnetgrid <- expand.grid(size=c(5, 10, 15, 20, 25, 30, 32, 35, 40, 43), 
                        decay=c(0.001, 0.01, 0.1), bag=FALSE)

# Creamos varios modelos de redes
clust <- parallel::makeCluster(6)
doParallel::registerDoParallel(cl = clust)
rednnet<- train(y~nr.employed+month.may+poutcome.success+poutcome.failure+contact.cellular+
                  month.nov+default.no+campaign+job.retired+emp.var.rate+education.university.degree+
                  month.apr+month.aug+education.unknown+cons.price.idx+marital.single+
                  job.services+job.blue_collar,
                data=bank,
                method="avNNet",
                maxit=100,
                trControl=control,
                tuneGrid=avnnetgrid)
foreach::registerDoSEQ()
parallel::stopCluster(clust)

rednnet



# Validacion cruzada repetida para los mejores modelos
red1 <- cruzadaavnnetbin(data=bank, vardep="y", 
                         listconti=c("nr.employed", "month.may", "poutcome.success", "poutcome.failure", "contact.cellular", 
                                     "month.nov", "default.no", "campaign", "job.retired", "emp.var.rate", 
                                     "education.university.degree", "month.apr", "month.aug", "education.unknown", 
                                     "cons.price.idx", "marital.single", "job.services", "job.blue_collar"),
                         listclass=c(""), grupos=4, sinicio=12345, repe=5,
                         size=c(5), decay=c(0.01), repeticiones=5, itera=200)
red1$modelo="5 - 0.01"

red2 <- cruzadaavnnetbin(data=bank, vardep="y", 
                         listconti=c("nr.employed", "month.may", "poutcome.success", "poutcome.failure", "contact.cellular", 
                                     "month.nov", "default.no", "campaign", "job.retired", "emp.var.rate", 
                                     "education.university.degree", "month.apr", "month.aug", "education.unknown", 
                                     "cons.price.idx", "marital.single", "job.services", "job.blue_collar"),
                         listclass=c(""), grupos=4, sinicio=12345, repe=5,
                         size=c(10), decay=c(0.01), repeticiones=5, itera=200)
red2$modelo="10 - 0.01"

red3 <- cruzadaavnnetbin(data=bank, vardep="y", 
                         listconti=c("nr.employed", "month.may", "poutcome.success", "poutcome.failure", "contact.cellular", 
                                     "month.nov", "default.no", "campaign", "job.retired", "emp.var.rate", 
                                     "education.university.degree", "month.apr", "month.aug", "education.unknown", 
                                     "cons.price.idx", "marital.single", "job.services", "job.blue_collar"),
                         listclass=c(""), grupos=4, sinicio=12345, repe=5,
                         size=c(32), decay=c(0.01), repeticiones=5, itera=200)
red3$modelo="32 - 0.01"

red4 <- cruzadaavnnetbin(data=bank, vardep="y", 
                         listconti=c("nr.employed", "month.may", "poutcome.success", "poutcome.failure", "contact.cellular", 
                                     "month.nov", "default.no", "campaign", "job.retired", "emp.var.rate", 
                                     "education.university.degree", "month.apr", "month.aug", "education.unknown", 
                                     "cons.price.idx", "marital.single", "job.services", "job.blue_collar"),
                         listclass=c(""), grupos=4, sinicio=12345, repe=5,
                         size=c(37), decay=c(0.1), repeticiones=5, itera=200)
red4$modelo="37 - 0.01"

red5 <- cruzadaavnnetbin(data=bank, vardep="y", 
                         listconti=c("nr.employed", "month.may", "poutcome.success", "poutcome.failure", "contact.cellular", 
                                     "month.nov", "default.no", "campaign", "job.retired", "emp.var.rate", 
                                     "education.university.degree", "month.apr", "month.aug", "education.unknown", 
                                     "cons.price.idx", "marital.single", "job.services", "job.blue_collar"),
                         listclass=c(""), grupos=4, sinicio=12345, repe=5,
                         size=c(43), decay=c(0.1), repeticiones=5, itera=200)
red5$modelo="43 - 0.01"
stopCluster(cl)


union1<-rbind(red1, red2, red3, red4, red5)
par(cex.axis=0.5)
boxplot(data=union1, tasa~modelo, main="Tasa de Fallos")
boxplot(data=union1, auc~modelo,  main="Curva ROC")




# Mejor modelo
set.seed(12345)
control<-trainControl(method="repeatedcv", number=4, repeats=2, savePredictions="all", classProbs=TRUE) 
# cambiar los valores 
avnnetgrid<-expand.grid(size=c(5), decay=c(0.01), bag=FALSE)

clust <- parallel::makeCluster(6)
doParallel::registerDoParallel(cl = clust)
avnnet<-train(y~nr.employed+month.may+poutcome.success+poutcome.failure+contact.cellular+month.nov+
                default.no+campaign+job.retired+emp.var.rate+education.university.degree+month.apr+
                month.aug+education.unknown+cons.price.idx+marital.single+job.services+job.blue_collar, 
              data=bank, method="avNNet", linout=FALSE, maxit=200, 
              repeats=2, trControl=control, tuneGrid=avnnetgrid)
foreach::registerDoSEQ()
parallel::stopCluster(clust)



# Curva Lift
library(data.table)
curva_lift = data.table(compra=avnnet$pred$obs, score=avnnet$pred$pred)
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



# **************************
# Validacion cruzada avNNet
# **************************

cruzadaavnnetbin<-function(data=data, vardep="vardep", 
                           listconti="listconti", listclass="listclass", 
                           grupos=4, sinicio=1234, repe=5, size=c(5), 
                           decay=c(0.01), repeticiones=5, itera=100) { 
  library(dummies)
  library(MASS)
  library(reshape)
  library(caret)
  library(dplyr)
  library(pROC)
  
  # PreparaciÃ³n del archivo
  
  # b)pasar las categÃ³ricas a dummies
  
  if (listclass!=c("")) {
    databis<-data[,c(vardep,listconti,listclass)]
    databis<- dummy.data.frame(databis, listclass, sep = ".")
  } else {
    databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center=means, scale=sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste(vardep,"~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method="repeatedcv", number=grupos, repeats=repe,
                        savePredictions="all", classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  avnnetgrid<-expand.grid(size=size, decay=decay, bag=FALSE)
  
  clust <- parallel::makeCluster(6)
  doParallel::registerDoParallel(cl = clust)
  avnnet<-train(y~nr.employed+month.may+poutcome.success+poutcome.failure+contact.cellular+
                  month.nov+default.no+campaign+job.retired+emp.var.rate+education.university.degree+
                  month.apr+month.aug+education.unknown+cons.price.idx+marital.single+
                  job.services+job.blue_collar, 
                data=databis, method="avNNet", linout=FALSE, maxit=itera, 
                repeats=repeticiones, trControl=control, tuneGrid=avnnetgrid)
  
  foreach::registerDoSEQ()
  parallel::stopCluster(clust)
  
  print(avnnet$results)
  
  preditest<-avnnet$pred
  
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
  
  medias<-preditest %>%
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
  
  mediasbis<-preditest %>%
    group_by(Rep) %>%
    summarize(auc=1*auc(obs, Yes))
  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
  return(medias)
  
}


## Ejemplo de utilización cruzada LOGISTICA Y AVNNET

# load ("c:/saheartbis.Rda")
# 
# medias1<-cruzadalogistica(data=saheartbis,
#  vardep="chd",listconti=c("sbp", "tobacco", "ldl",
#   "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""), grupos=4,sinicio=1234,repe=5)
# 
#  medias1$modelo="Logística"
# 
# 
# medias2<-cruzadaavnnetbin(data=saheartbis,
#  vardep="chd",listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),grupos=4,sinicio=1234,repe=5,
#   size=c(5),decay=c(0.1),repeticiones=5,itera=200)
# 
#   medias2$modelo="avnnet"
# 
# union1<-rbind(medias1,medias2)
# 
# par(cex.axis=0.5)
# boxplot(data=union1,tasa~modelo,main="TASA FALLOS")
# boxplot(data=union1,auc~modelo,main="AUC")
# 
# 
# 
