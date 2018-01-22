devtools::install_github("cpsievert/LDAvisData")
data(Reviews, package = "LDAvisData")

url <- "http://cran.us.r-project.org/src/contrib/Archive/cldr/cldr_1.1.0.tar.gz"
pkgFile<-"cldr_1.1.0.tar.gz"
download.file(url = url, destfile = pkgFile)
install.packages(pkgs=pkgFile, type = "source", repos = NULL)
unlink(pkgFile)

install.packages("lda")
install.packages("LDAvis")
install.packages("servr")

library(tm)
library(lda)
library(LDAvis)
library(servr)
library(text2vec)

library(textstem)


#read file by navigation
Final_TW_Newest<- read.csv(file.choose(),encoding = "UTF-8")

###get key, tweets, time of TW dataset
Final_TW_Tweets<- Final_TW_Newest[c(1,9,2,7)]
Final_TW_Tweets<- subset(Final_TW_Tweets, Language=='eng')
Final_TW_Tweets<- data.frame(na.omit(Final_TW_Tweets))
colnames(Final_TW_Tweets)<- c("key", "Text","time","language")

#remove non-english posts
Final_TW_Tweets_lag<- detectLanguage(Final_TW_Tweets[[2]])
Final_TW_Tweets<- cbind(Final_TW_Tweets,Final_TW_Tweets_lag)
Final_TW_Tweets<- subset(Final_TW_Tweets,detectedLanguage=="ENGLISH")[c(1,2,3)]




# read in some stopwords:

stop_words <- stopwords("SMART")
word_bag<- c("â???s","johnson","rt","ed","fc","â","pa","sta","cdc")
stop_words<- c(stop_words, word_bag)

removeURL <- function (sentence){
  #convert to lower-case 
  sentence <- tolower(sentence)
  removeURL <- function(x) gsub('(http.*) |(https.*) |(http.*)$|\n|ã|"|â???s|â|johnson', "", x)
  sentence <- removeURL(sentence)
}

clean <- function (sentence){
  remove <- function(x) gsub('wh |ã???|ãs|â???s|Å¡ã£|ã¥|ã£Æ’ã¦|iã£Æ’ã¦|rt | ed| fc| bd| bc|wh |ba | ce | ar | wn | ne | it | ae | bb | fef | di | ale | ee | gt | ra | dr | s | d |cf |bf | cf|af | st | amp | ,|, ', "", x)
  sentence <- remove(sentence)
}


# pre-processing:
Final_TW_Tweets$Text <- sapply(Final_TW_Tweets$Text, function(x) removeURL(x))
Final_TW_Tweets$Text <- removeNumbers(Final_TW_Tweets$Text)
Final_TW_Tweets$Text <- lemmatize_strings(Final_TW_Tweets$Text)
Final_TW_Tweets$Text <- removeWords(Final_TW_Tweets$Text, stop_words)
Final_TW_Tweets$Text <- gsub("'", "", Final_TW_Tweets$Text)  # remove apostrophes
Final_TW_Tweets$Text <- gsub("[[:punct:]]", " ", Final_TW_Tweets$Text)  # replace punctuation with space
Final_TW_Tweets$Text <- sapply(Final_TW_Tweets$Text, function(x) clean(x))
Final_TW_Tweets$Text <- gsub("[[:cntrl:]]", "", Final_TW_Tweets$Text)  # replace control characters with space
Final_TW_Tweets$Text <- gsub("^[[:space:]]+", "", Final_TW_Tweets$Text) # remove whitespace at beginning of documents
Final_TW_Tweets$Text <- gsub("[[:space:]]+$", "", Final_TW_Tweets$Text) # remove whitespace at end of documents


# tokenize on space and output as a list:
doc.list <- strsplit(Final_TW_Tweets$Text, "[[:space:]]+")
#doc.list <- strsplit(Final_TW_Tweets$Text, " ")

#doc.list1<- list(weightMatrix[1,])

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

#termtable<- as.data.frame(term.table)

# remove terms that are stop words or occur fewer than 5 times:
#del <- names(term.table) %in% stop_words | term.table < 800
del <- term.table < 801
term.table <- term.table[!del]
vocab <- names(term.table)

vocab1<- dimnames(weightMatrix)[1]
vocab1<- as.character(unlist(vocab1))

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)



# Compute some statistics related to the data set:
D <- length(documents)  # number of documents
W <- length(vocab)  # number of terms in the vocab 
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document 
N <- sum(doc.length)  # total number of tokens in the data
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus

# MCMC and model tuning parameters:
K <- 20
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:
#set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 2 minutes on laptop

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))
datasetReviews <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)


# create the JSON object to feed the visualization:
json <- createJSON(phi = datasetReviews$phi, 
                   theta = datasetReviews$theta, 
                   doc.length = datasetReviews$doc.length, 
                   vocab = datasetReviews$vocab, 
                   term.frequency = datasetReviews$term.frequency)


topic_list_LDA_R<- top.topic.words(fit$topics, num.words = 15, by.score = FALSE)


serVis(json, out.dir = 'viiiiiiiiii', open.browser = TRUE)

sldaem<- slda.em(documents=documents, 
                 K=K, 
                 vocab=vocab, 
                num.e.iterations=G, 
                num.m.iterations=G, 
                alpha, 
                eta, 
                annotations, 
                params, 
                variance, 
                logistic = FALSE, 
                lambda = 10, 
                regularise = FALSE, 
                method = "sLDA", 
                trace = 0L, 
                MaxNWts=3000)


prediction<- slda.predict(documents, fit$topics, model, alpha, eta, num.iterations = 100, average.iterations = 50, trace = 0L)


pridicDoc<- slda.predict.docsums(documents=documents, 
                                    fit$topics, 
                                    alpha, 
                                    eta, 
                                    num.iterations = 10000, 
                                    average.iterations = 5000, 
                                    trace = 0L)

getTopTopic<- function(pridicDoc){
  
  t<- list()
  for(i in 1:dim(pridic)[2]){
    
    for(j in 1:dim(pridic)[1]){
      
      if(pridic[j,i]==max(pridic[,i])){
        
        t<- cbind(t,j)
        break
      }
    }
  }
  
  return(t)
}

# apply the function and get topic assignment list
topic<- getTopTopic(pridic)

topic<- as.data.frame(t(topic))

# Combine the post dataset with topic
Final_TW_Tweets_Topic<- cbind(Final_TW_Tweets, topic)
colnames(Final_TW_Tweets_Topic)[5]<- "TopicID"
# link TopicID with topic Content
Final_TW_Tweets_Topic_Final<- merge(x = Final_TW_Tweets_Topic, y = topic_list_LDA, by = "TopicID", all.x = TRUE)







assignment_list_LDA_R<- top.topic.words(t(pridicDoc), num.words = 2, by.score = FALSE)

topdocs<- top.topic.documents(fit$document_sums, num.documents = 100, alpha = 0.1)

x<- predictive.distribution(fit$document_sums, fit$topics, alpha, eta)
