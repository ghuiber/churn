# 20140407
#
# try and replicate Eric Chiang's Python code of 
# churn model published at blog.yhathq.com

# main settings: clean up, set output options, require libraries
rm(list=ls(all=TRUE))
require(e1071)        # for SVM
require(FNN)          # for faster KNN than class::knn
require(randomForest) # for RF

data <- read.csv('GitHub/churn/data/churn.csv')

# dependent variable
y <- data$Churn.=='True.'

# now set covariates
X <- subset(data,select=-c(State,Area.Code,Phone,Churn.))
X[c('Int.l.Plan','VMail.Plan')] <- X[c('Int.l.Plan','VMail.Plan')]=='yes'
X.scaled <- scale(X)

# compare 3 classifiers with k-fold cross-validation
modelBakeOff <- function(k=5) {
    dat <- data.frame(X.scaled,y=as.factor(y))
    set.seed(20140407)
    folds=sample(rep(1:k,length=nrow(dat)))
    acc  <- matrix(NA,k,3)
    # try some classifiers, as close as I can get them to sklearn defaults
    for(j in 1:k) {
        # SVM classifier
        svmfit <- svm(y~.,data=dat[folds!=j,],kernel="radial",scale=FALSE)
        svmerr <- predict(svmfit, newdata=dat[folds==j,])
        acc[j,1] <- sum(svmerr==dat[folds==j,]$y)/nrow(dat[folds==j,])
        # randomForest
        train  <- folds!=j
        rffit  <- randomForest(y~.,data=dat,subset=train,ntree=10)
        rferr  <- predict(rffit,dat[folds==j,])
        acc[j,2] <- sum(rferr==dat[folds==j,]$y)/nrow(dat[folds==j,])
        # K-nearest neighbors
        knnfit <- knn(X.scaled[folds!=j,],X.scaled[folds==j,],cl=factor(y)[folds!=j],k=5)
        acc[j,3] <- sum(knnfit==factor(y)[folds==j])/length(knnfit)        
    }
    colnames(acc) <- c('SVM','rF','KNN')
    return(acc)
}
svmacc <- modelBakeOff()

