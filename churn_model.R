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
    # home of the accuracy rates
    acc   <- matrix(NA,k,3)
    # homes of counts for confusion matrices
    cmsvm <- list()
    cmrf  <- list()
    cmknn <- list()
    # homes for probability tables
    probsvm <- list()
    probrf  <- list()
    probknn <- list()
    # try some classifiers, as close as I can get them to sklearn defaults
    for(j in 1:k) {
        # SVM classifier
        svmfit <- svm(y~.,data=dat[folds!=j,],kernel="radial",scale=FALSE,probability=TRUE)
        svmerr <- predict(svmfit, newdata=dat[folds==j,])
        svmprob <- predict(svmfit, newdata=dat[folds==j,], probability=TRUE)
        acc[j,1] <- sum(svmerr==dat[folds==j,]$y)/nrow(dat[folds==j,])
        cmsvm[[j]] <- table(svmerr,dat[folds==j,]$y)
        probsvm[[j]] <- cbind(svmerr,attr(svmprob,'probabilities')[,2])
        
        # randomForest
        train  <- folds!=j
        rffit  <- randomForest(y~.,data=dat,subset=train,ntree=10)
        rferr  <- predict(rffit,dat[folds==j,])
        rfprob <- predict(rffit,dat[folds==j,],type='prob')
        acc[j,2] <- sum(rferr==dat[folds==j,]$y)/nrow(dat[folds==j,])
        cmrf[[j]] <- table(rferr,dat[folds==j,]$y)
        probrf[[j]] <- cbind(rferr,rfprob[,2])
        
        # K-nearest neighbors
        knnfit <- knn(X.scaled[folds!=j,],X.scaled[folds==j,],cl=factor(y)[folds!=j],k=5,prob=TRUE)
        acc[j,3] <- sum(knnfit==factor(y)[folds==j])/length(knnfit)
        cmknn[[j]] <- table(knnfit,dat[folds==j,]$y)
        probknn[[j]] <- cbind(knnfit,attr(knnfit,'prob'))
    }
    colnames(acc) <- c('SVM','rF','KNN')
    # now add the k tables together for full 
    # cross-validated confusion matrices
    cmsvm <- Reduce("+",cmsvm)
    cmrf  <- Reduce("+",cmrf)
    cmknn <- Reduce("+",cmknn)
    dimnames(cmsvm)$svmerr <- paste('predicted',dimnames(cmsvm)$svmerr,sep='.')
    dimnames(cmrf)$rferr   <- paste('predicted',dimnames(cmrf)$rferr,sep='.')
    dimnames(cmknn)$knnfit <- paste('predicted',dimnames(cmknn)$knnfit,sep='.')
    # now rbind() together the k probability matrices
    # for full cross-validated probability estimates
    probsvm <- Reduce("rbind",probsvm)
    probrf  <- Reduce("rbind",probrf)
    probknn <- Reduce("rbind",probknn)    
    
    cm <- list(cmsvm,cmrf,cmknn)
    prob <- list(probsvm,probrf,probknn)
    names(cm) <- colnames(acc)
    names(prob) <- colnames(acc)
    
    out <- list(acc,cm,prob)
    names(out) <- c('CV Accuracy','CV Confusion Matrices','CV Probabilities')
    return(out)
}
bakeoff <- modelBakeOff()

print('')
print('Cross-validated accuracy rates')
print(bakeoff[[1]])
print('')
print('Cross-validated confusion matrices')
print(bakeoff[[2]])
