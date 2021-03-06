#' @export
aggregate_graphs <- function(cgraphs, method = c("frequentist", "bayesian"), df = NULL)
{
  if(!is.list(cgraphs))
    stop("dags is not as list")
  if (length(cgraphs) == 1)
    stop("dags is of length 1")

  base <- cgraphs[[1]]
  # see if all the graphs have the EXACT same nodes
  same_nodes <- lapply(cgraphs, function(cgraph) {
    isTRUE(all.equal(base$nodes, cgraph$nodes))
  })
  same_nodes <- isTRUE(all.equal(unlist(same_nodes), rep(T, length(cgraphs))))
  if (!same_nodes)
    stop("Not all the graphs have the same nodes")

  method <- match.arg(method)
  bs.weights <- rep(0, length(cgraphs))
  if (method == "frequentist") {
    bs.weights <- rep(1, length(cgraphs))
  }
  if (method == "bayesian") {
    df <- as.data.frame(lapply(df, function(x) { (x - mean(x))/sd(x) }))
    for (i in 1:length(cgraphs)) {
      graph <- as.dag(cgraphs[[i]])
      bs.weights[i] <- score_graph(graph, df)
    }
    bs.weights <- exp(-.5*(bs.weights - min(bs.weights)))
  }
  bs.weights <- round(bs.weights, digits = 5)
  cgraphs <- lapply(cgraphs, function(cgraph) {
    .prepare_cgraph_for_call(cgraph, F, T, T)
  })

  table <- .Call("cf_aggregate_cgraphs", cgraphs, bs.weights)
  table <- as.data.frame(table)

  cgraph <- cgraphs[[1]]
  table[[1]] <- cgraph$nodes[table[[1]]]
  table[[2]] <- cgraph$nodes[table[[2]]]
    names(table) <- c("node1","node2", "<--", "---", "-->", "<~~",
                      "~~>", "<++", "++>","<-o", "o->", "<->", "o-o")

  table <- table[, c(T, T, colSums(table[, -(1:2)]) != 0)]

  output <- list(nodes = cgraph$nodes, table = table)
  class(output) <- c("aggregated-cgraphs")
  return(output)
}
#' @export
vote <- function(agg_pdags, threshold = .5, method = c("plurality", "majority",
                  "relative_majority", "square_relative_majority"))
{
  plurality <- function(x) {
    max <- max(x)
    n_max <- 0
    for (value in x) {
      if (value == max) {
        n_max <- n_max + 1
      }
    }
    if (n_max > 1)
      return(0)
    else
      return(match(max, x))
  }

  majority <- function(x) {
    for(i in 1:length(x) ) {
      if (x[i] > .5)
        return(i)
    }
    return(0)
  }

  relative_majority <- function(x) {
    for (i in 1:length(x)) {
      if (x[i] > sum(x[-i]))
        return(i)
    }
    return(0)
  }

  square_relative_majority <- function(x) {
    for (i in 1:length(x)) {
      if (x[i]^2 > sum(x[-i]^2))
        return(i)
    }
    return(0)
  }

  method <- match.arg(method)

  voting_method <- switch (method,
                           "plurality"                = plurality,
                           "majority"                 = majority,
                           "relative_majority"        = relative_majority,
                           "square_relative_majority" = square_relative_majority
  )

  calculate_edge <- function(src, dst, x) {
    # these need to be chars because R is dumb
    return(switch(as.character(x),
                 "0"  = c(src, dst, "---"),
                 "1"  = c(dst, src, "-->"),
                 "2"  = c(src, dst, "---"),
                 "3"  = c(src, dst, "-->"),
                 "4"  = c(dst, src, "~~>"),
                 "5"  = c(src, dst, "~~>"),
                 "6"  = c(src, dst, "++>"),
                 "7"  = c(src, dst, "++>"),
                 "8"  = c(dst, src, "o->"),
                 "9"  = c(src, dst, "o->"),
                 "10" = c(src, dst, "<->"),
                 "11" = c(src, dst, "o-o")

    ))
  }

  df <- agg_pdags$table
  df <- df[rowSums(df[, -c(1:2)]) > threshold,]
  nodes <- agg_pdags$nodes
  n_edges <- nrow(df)
  if (n_edges == 0) {
    warning("Threshold too high, resulting graph is empty! Returning NA")
    return(NA)
  }
  edges <- matrix(data = "", nrow = n_edges, ncol = 3)
  for (i in 1:n_edges) {
    edges[i,] <- calculate_edge(df[i,1], df[i,2], voting_method(df[i, -c(1:2)]))
  }
  return(cgraph(nodes, edges))
}

#' @export
vote2 <- function(agg_pdags) {
  df <- agg_pdags$table
  df$'!' <- 1- rowSums(df[, -(1:2)])

  plurality <- function(x) {
    max <- max(x)
    n_max <- 0
    for (value in x) {
      if (value == max) {
        n_max <- n_max + 1
      }
    }
    if (n_max > 1)
      return(0)
    else
      return(match(max, x))
  }


  edges <- c()
  for (i in 1:nrow(df)) {
    edge_type <- plurality(df[i, -c(1:2)])
    if (edge_type == 0 || edge_type == 2)
      edges <- c(edges, df[i, 1], df[i, 2], "---" )
    else if(edge_type == 1)
      edges <- c(edges, df[i, 2], df[i, 1], "-->")
    else if(edge_type == 3)
      edges <- c(edges, df[i, 1], df[i, 2], "-->" )
  }
  nodes <- agg_pdags$nodes
  edges <- matrix(edges, ncol = 3, byrow = T)
  return(cgraph(nodes, edges))
}

#' @export
votek<- function(agg_pdags, k) {
  df <- agg_pdags$table
  df$'!' <- 1- rowSums(df[, -(1:2)])/k

  plurality <- function(x) {
    max <- max(x)
    n_max <- 0
    for (value in x) {
      if (value == max) {
        n_max <- n_max + 1
      }
    }
    if (n_max > 1)
      return(0)
    else
      return(match(max, x))
  }


  edges <- c()
  for (i in 1:nrow(df)) {
    edge_type <- plurality(df[i, -c(1:2)])
    if (edge_type == 0 || edge_type == 2)
      edges <- c(edges, df[i, 1], df[i, 2], "---" )
    else if(edge_type == 1)
      edges <- c(edges, df[i, 2], df[i, 1], "-->")
    else if(edge_type == 3)
      edges <- c(edges, df[i, 1], df[i, 2], "-->" )
  }
  nodes <- agg_pdags$nodes
  edges <- matrix(edges, ncol = 3, byrow = T)
  return(cgraph(nodes, edges))
}
