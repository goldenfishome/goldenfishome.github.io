library(rlang)
library(usethis)
library(devtools)
library(base64enc)
library(RCurl)
library(networkD3)
library(arules)
library(jsonlite)
library(streamR)
library(rjson)
library(tokenizers)
library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
library(syuzhet)
library(stringr)
library(arulesViz)
library(igraph)


setwd("C:/Users/yujia/Desktop/Georgetown/501/project/ARM")
df <- read.csv("ARMCleaned.csv")
colnames(df)[1] <-"OverallStatus"

str(df)

df<-df %>%
  mutate_all(as.character)

#colnames(df) <- NULL

#write.csv(df,"armRecord.csv",row.names = F)
write.csv(df,"armRecord.csv",row.names = F)

myData <- read.transactions("armRecord.csv",
                             rm.duplicates = F,
                             format="basket",
                             sep=",")
inspect(myData)

record_rules <- arules::apriori(myData,parameter = list(support=.005,
                                                  confidence=0.01,
                                                  minlen=2),
                                appearance = list(lhs=c('shortLen', 'relatively_shortLen', 
                                                        'medianLen', 'longLen'),
                                                  default="rhs"))

inspect(record_rules)
##  SOrt by Conf
SortedRules_conf <- sort(record_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf[1:20])
## Sort by Sup
SortedRules_sup <- sort(record_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup[1:20])
#inspect(SortedRules_sup)
## Sort by Lift
SortedRules_lift <- sort(record_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift[1:20])

tr.rules <-SortedRules_lift[1:50]

#######################################################
########  Using NetworkD3 To View Results   ###########
#######################################################

Rules_tr2<-DATAFRAME(tr.rules, separate = TRUE)
(head(Rules_tr2))
str(Rules_tr2)


## Convert to char
Rules_tr2$LHS<-as.character(Rules_tr2$LHS)
Rules_tr2$RHS<-as.character(Rules_tr2$RHS)

## Remove all {}
Rules_tr2[] <- lapply(Rules_tr2, gsub, pattern='[{]', replacement='')
Rules_tr2[] <- lapply(Rules_tr2, gsub, pattern='[}]', replacement='')

Rules_tr2

## Other options for the following
#Rules_Lift<-Rules_DF2[c(1,2,5)]
#Rules_Conf<-Rules_DF2[c(1,2,4)]
#names(Rules_Lift) <- c("SourceName", "TargetName", "Weight")
#names(Rules_Conf) <- c("SourceName", "TargetName", "Weight")
#head(Rules_Lift)
#head(Rules_Conf)

###########################################
###### Do for SUp, Conf, and Lift   #######
###########################################
## Remove the sup, conf, and count
## USING LIFT
Rules_L<-Rules_tr2[c(1,2,6)]
names(Rules_L) <- c("SourceName", "TargetName", "Weight")
head(Rules_L,30)

## USING SUP
Rules_S<-Rules_tr2[c(1,2,3)]
names(Rules_S) <- c("SourceName", "TargetName", "Weight")
head(Rules_S,30)

## USING CONF
Rules_C<-Rules_tr2[c(1,2,4)]
names(Rules_C) <- c("SourceName", "TargetName", "Weight")
head(Rules_C,30)

## CHoose and set
#Rules_Sup<-Rules_C
Rules_Sup<-Rules_L
#Rules_Sup<-Rules_S

###########################################################################
#############       Build a NetworkD3 edgeList and nodeList    ############
###########################################################################

#edgeList<-Rules_Sup
# Create a graph. Use simplyfy to ensure that there are no duplicated edges or self loops
#MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE))
#plot(MyGraph)

############################### BUILD THE NODES & EDGES ####################################

# build for duration data
(edgeList<-Rules_Sup)
nrow(edgeList)
(MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE)))

nodeList <- data.frame(id = c(0:(igraph::vcount(MyGraph) - 1)), 
                       # because networkD3 library requires IDs to start at 0
                       label = igraph::V(MyGraph)$name)
## Node Degree
(nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                       v = igraph::V(MyGraph), mode = "all")))

## Betweenness
BetweenNess <- igraph::betweenness(MyGraph, 
                                   v = igraph::V(MyGraph), 
                                   directed = TRUE) 

(nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))

## This can change the BetweenNess value if needed
#BetweenNess<-BetweenNess/100

## For scaling...divide by 
## RE:https://en.wikipedia.org/wiki/Betweenness_centrality
##/ ((igraph::vcount(MyGraph) - 1) * (igraph::vcount(MyGraph)-2))
## For undirected / 2)
## Min-Max Normalization
##BetweenNess.norm <- (BetweenNess - min(BetweenNess))/(max(BetweenNess) - min(BetweenNess))


## Node Degree



###################################################################################
########## BUILD THE EDGES #####################################################
#############################################################
# Recall that ... 
# edgeList<-Rules_Sup
getNodeID <- function(x){
  which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
}
## UPDATE THIS !! depending on # choice
(getNodeID("shortLen")) 

edgeList <- plyr::ddply(
  Rules_Sup, .variables = c("SourceName", "TargetName" , "Weight"), 
  function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                          TargetID = getNodeID(x$TargetName)))

head(edgeList)
nrow(edgeList)

########################################################################
##############  Dice Sim ################################################
###########################################################################
#Calculate Dice similarities between all pairs of nodes
#The Dice similarity coefficient of two vertices is twice 
#the number of common neighbors divided by the sum of the degrees 
#of the vertices. Method dice calculates the pairwise Dice similarities 
#for some (or all) of the vertices. 
DiceSim <- igraph::similarity.dice(MyGraph, vids = igraph::V(MyGraph), mode = "all")
head(DiceSim)

#Create  data frame that contains the Dice similarity between any two vertices
F1 <- function(x) {data.frame(diceSim = DiceSim[x$SourceID +1, x$TargetID + 1])}
#Place a new column in edgeList with the Dice Sim
head(edgeList)
edgeList <- plyr::ddply(edgeList,
                        .variables=c("SourceName", "TargetName", "Weight", 
                                     "SourceID", "TargetID"), 
                        function(x) data.frame(F1(x)))
head(edgeList)

##################################################################################
##################   color #################################################
######################################################
# COLOR_P <- colorRampPalette(c("#00FF00", "#FF0000"), 
#                             bias = nrow(edgeList), space = "rgb", 
#                             interpolate = "linear")
# COLOR_P
# (colCodes <- COLOR_P(length(unique(edgeList$diceSim))))
# edges_col <- sapply(edgeList$diceSim, 
#                     function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
# nrow(edges_col)

## NetworkD3 Object
#https://www.rdocumentation.org/packages/networkD3/versions/0.4/topics/forceNetwork

D3_network_records <- networkD3::forceNetwork(
  Links = edgeList, # data frame that contains info about edges
  Nodes = nodeList, # data frame that contains info about nodes
  Source = "SourceID", # ID of source node 
  Target = "TargetID", # ID of target node
  Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
  NodeID = "label", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
  Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
  Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
  height = 700, # Size of the plot (vertical)
  width = 900,  # Size of the plot (horizontal)
  fontSize = 15, # Font size
  linkDistance = networkD3::JS("function(d) { return d.value*100; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
  #linkWidth = networkD3::JS("function(d) { return d.value*5; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
  opacity = 5, # opacity
  zoom = TRUE, # ability to zoom when click on the node
  opacityNoHover = 5, # opacity of labels when static
  #linkColour = "green"   ###"edges_col"red"# edge colors
) 

# Plot network


# Save network as html file
networkD3::saveNetwork(D3_network_records, 
                       "NetD3_DurationDays.html", selfcontained = TRUE)

networkD3::saveNetwork(D3_network_records, 
                       "NetD3_days_long.html", selfcontained = TRUE)


###########################################################################
#############       Build a visNetwork edgeList and nodeList    ############
###########################################################################
### for duration data
lenNode <- data.frame(id=nodeList$id,label=nodeList$label)
lenEdge <- data.frame(from=edgeList$SourceID,to=edgeList$TargetID,
                      weight=edgeList$Weight)

lenEdge
visNetwork(lenNode, lenEdge, layout = "layout_with_fr",
           arrows="middle")

visNetwork(lenNode, lenEdge) %>% 
  visIgraphLayout(layout = "layout_with_fr") %>% 
  visEdges(arrows = "middle") %>%
  visNodes(font = list(size=30),
           color = "#1f7f38")

####################
### igraph
###########

(igraph1 <- 
   graph_from_data_frame(d = lenEdge, vertices = lenNode, directed = TRUE))

E(igraph1)
E(igraph1)$weight

E_Weight<-lenEdge$weight
E(igraph1)$weight <- edge.betweenness(igraph1)
E(igraph1)$color <- "green"

V(igraph1)$size = 5

layout1 <- layout.fruchterman.reingold(igraph1)
#plot(igraph1)

plot(igraph1, edge.arrow.size = 0.3,
     #vertex.size=E_Weight*5,
     vertex.color="lightblue",
     layout=layout1,
     edge.arrow.size=.5,
     vertex.label.cex=0.8, 
     vertex.label.dist=2, 
     #edge.curved=0.2,
     #vertex.label.color="black",
     edge.weight=5, 
     #edge.width=E(My_igraph2)$weight*10,
     #edge_density(My_igraph2),
     ## Affect edge lengths
     #rescale = T, 
     ylim=c(-0.5,0.6),
     xlim=c(-1,1.1)
)
