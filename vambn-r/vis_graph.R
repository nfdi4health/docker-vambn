library(ggraph)
library(bnlearn)
library(igraph)
library(tidygraph)

args <- commandArgs(trailingOnly = TRUE)

inputFile <- ifelse(length(args) >= 1, args[1], "reports/bn/modulary_altoidaNewADPT_equal/bootstrap.rds")
outputFile <- ifelse(length(args) >= 2, args[2], "test2.png")

arcs <- readRDS(inputFile)
A <- arcs[(arcs$strength > 0.4) & (arcs$direction >= 0.4), ]
IG <- graph_from_edgelist(as.matrix(A[, c("from", "to")]))
E(IG)$weight <- A$strength

node_names <- V(IG)
filtered_nodes <- node_names[grep("_z0$|^SA_", node_names)]
new_node_names <- gsub("SA_|_VIS1|_z0", "", filtered_nodes)
name_mapping <- setNames(filtered_nodes, new_node_names)

IG <- delete.vertices(IG, grepl("_s", names(V(IG))))
V(IG)

process_text <- function(x){
    x <- gsub("_VIS1", "", x)
    x <- gsub("_z0", "", x)
    x <- gsub("SA_", "", x)
    x <- gsub("group", "DX", x)
    return(x)
}

get_labels <- function(x){
    if (grepl("MMSE", x)) {
        return("Clinical")
    } else if (grepl("age|gender|education_years|group|DX", x)) {
        return("Demographics")
    } else  if (grepl("SA", x)) {
        return("Cognitive Domain Scores")
    } else {
        return("RMT Data Modules")
    }
}

plot_bnlearn_graph <- function(igraph_object){
  tidygraph_object <- as_tbl_graph(igraph_object)

  node.labels <- sapply(attr(V(igraph_object), "names"), get_labels)
  node.names <- sapply(attr(V(igraph_object), "names"), process_text)
  edge.weight <- E(igraph_object)$weight

  return(ggraph(tidygraph_object, layout = 'fr', maxiter=1000) +
    geom_edge_link(aes(label=round(edge.weight*100, 2)), color="#c2c2c2",arrow=arrow(length=unit(2, "mm"), angle=38, type="closed"),
                width=1, check_overlap = TRUE, end_cap = circle(0.4, 'cm'),
                label_alpha = 0.6, label_size=2.5) +
    geom_node_point(aes(color=node.labels), size=9, alpha=.3) +
    geom_node_text(aes(label=node.names), color="#3a3939", repel=TRUE, size=3.5) +
    labs(color="Node Type") +
    theme_graph() +
    theme(legend.position = "bottom")
  )
}

g <- plot_bnlearn_graph(IG)
ggsave(outputFile, plot=g, width=23, height=16, units="cm")
