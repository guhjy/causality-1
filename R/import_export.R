

export_bnlearn_object_to_tetrad <- function(file, model) {
  # This function exports an object of bn-class in a TETRAD compatible form
  write("Graph Nodes:", file = file, append = F)
  cat(names(model$nodes), file = file, append = T, sep = ",")
  write('\n', file = file, append = T)
  write("Graph Edges:", file = file, append = T)

  n <- 1
  for (i in 1:length(model$nodes)) {
    node <- names(model$nodes)[i]
    for (child in model$nodes[[i]]$children) {
      write(sprintf("%i. %s --> %s", n, node, child), file = file, append = T)
      n <- n + 1
    }
  }
  cat('\n', file = file, append = T)
}
#' @export
import_from_tetrad_file <- function(file, type = "cgraph", sort = F) {
  if (!file.exists(file))
    stop(sprintf("file does not exist!\n"))
  if(!is.logical(sort))
    stop("stop does not take on a boolean value!")

  # this function imports a tetrad graph
  # parse the tetrad file as a list of character vectors
  tmp_file <- read_lines(file)
  # get the names of the nodes from
  nodes <- strsplit(tmp_file[2], ",")[[1]]
  if (sort)
    nodes <- sort(nodes)
  # strip out first 4 lines and last line
  tmp_file <- tmp_file[5:( length(tmp_file) - 1)]
  # remove the number of the edge
  tmp_file <- sub("[0-9]+.", "", tmp_file)
  # prep the lines so the edge matrix can be filled in easily
  tmp_file <- strsplit(trimws(tmp_file), " ")
  # get the number of edges
  n_edges <- length(tmp_file)
  # instantiant edge matrix
  edges <- matrix(nrow = n_edges, ncol = 3)
  # fill it in!
  for (i in 1:n_edges) {
    # if edge is of type <--, change it to --> and warn the user
    if(tmp_file[[i]][2] == "<--") {
      warning(sprintf("edge %i is of type '<--' in %s; converting to type '-->", i, file) )
      edges[i, 1] <- tmp_file[[i]][3]
      edges[i, 2] <- tmp_file[[i]][1]
      edges[i, 3] <- "-->"
    } else {
      edges[i, 1] <- tmp_file[[i]][1]
      edges[i, 2] <- tmp_file[[i]][3]
      edges[i, 3] <- tmp_file[[i]][2]
    }
  }



  tmp_cgraph <- cgraph(nodes, edges)
  # check to see if input is a legal cgraph object
  if(is_valid_cgraph(tmp_cgraph) == FALSE)
    stop("imported graph is not a valid cgraph object")
  # if type filed isn't empty, attempt to parse it as the proposed type
  if(!is.null(type)) {
    if(type == "dag") {
      tmp_dag <- tryCatch(as.dag(tmp_cgraph), error = function(e) {message(paste(e)); NULL})
      if(is.null(tmp_dag))
         stop(sprintf("imported object is not a valid dag,
                      so it cannot be imported as a dag.
                      call the function with type = NULL to import it"
                      ))
      else
        return(tmp_dag)
    }
    if(type == "cgraph") {
      return(tmp_cgraph)
    }
  }
  else
    return(tmp_cgraph)
}
