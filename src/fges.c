#include <causality.h>
#include <cgraph.h>
#include <heap.h>
#include <dataframe.h>
#include <scores.h>
#include <pdx.h>
#include <local_meek.h>
#include <chickering.h>
#include <edgetypes.h>
#include <stdint.h>

//#define DEBUG 1

#ifndef DEBUG
#define DEBUG 0
#endif

struct gesrec {
    int   x;
    int   y;
    int   set_size;
    int   naxy_size;
    int  *set;
    int  *naxy;
}; /*  32 bytes */

struct cgraph * ccf_fges(struct dataframe df, score_func score, double *fargs,
                                              int *iargs);

struct dataframe calculate_df(SEXP Df, SEXP States) {
    struct dataframe data;
    data.nvar   = length(Df);
    data.nobs   = length(VECTOR_ELT(Df, 0));
    data.states = INTEGER(States);
    data.df     = malloc(data.nvar * sizeof(void *));
    for (int i = 0; i < data.nvar; ++i) {
        SEXP Df_i = VECTOR_ELT(Df, i);
        if (data.states[i])
            data.df[i] = INTEGER(Df_i);
        else
            data.df[i] = REAL(Df_i);
    }
    return data;
}

SEXP ccf_fges_wrapper(SEXP Df, SEXP ScoreType, SEXP States,
                 SEXP FloatingArgs, SEXP IntegerArgs)
{
    /*
    * calcluate the integer arguments and floating point arguments for the
    * score function.
    */
    int *iargs = NULL;
    if (!isNull(IntegerArgs))
        iargs = INTEGER(IntegerArgs);
    double *fargs = NULL;
    if (!isNull(FloatingArgs))
        fargs = REAL(FloatingArgs);

    /* populate df with the pointers to the columns of the R dataframe */

    struct dataframe data = calculate_df(Df, States);
    score_func score = NULL;
    // if (!strcmp(CHAR(STRING_ELT(ScoreType, 0)), BIC_SCORE))
    //     score = bic_score;
    // else if (!strcmp(CHAR(STRING_ELT(ScoreType, 0)), BDEU_SCORE))
    //     score = bdeu_score;
    // else
    //     error("nope\n");
    /*
    * All the preprocessing work has now been done, so lets instantiate
    * an empty graph and run FGES
    */
    struct cgraph *cg = ccf_fges(data, score, fargs, iargs);
    /* POST PROCESSING */
    free_cgraph(cg);
    free(data.df);
    return ScalarReal(0);
}


 int is_clique(struct cgraph *cg, int *nodes, int n_nodes)
{
    for (int i = 0; i < n_nodes; ++i) {
        int inode = nodes[i];
        for (int j = 0; j < i; ++j) {
            if (!adjacent_in_cgraph(cg, inode, nodes[j])) {
                return 0;
            }
        }
    }
    return 1;
}

int unblocked_semidirected_path(struct cgraph *cg, int src, int dst, int *onion,
                                            int onion_size)
{

    int unblocked_path = 0;
    struct ill *stack  = ill_insert_front(NULL, src, 0);
    int *marked = calloc(cg->n_nodes, sizeof(int));
    if (marked == NULL)
        error("Failed to allocate memory for marked in fges!\n");
    while (stack) {
        /* pop */
        int node = stack->key;
        struct ill *tmp = stack->next;
        free(stack);
        stack = tmp;
        /* check to if T \cup NAXY block the path. */
        for (int i = 0; i < onion_size; ++i) {
            if (node == onion[i])
                goto NEXT;
        }
        /* If node == dst, we have a cycle */
        if (node == dst) {
            unblocked_path = 1;
            goto CLEANUP;
        }
        /*
         * if we haven't visited the node before, we need to push all of its
         * spouses and children onto the stack and mark it
         */
        if (!marked[node]) {
            marked[node] = 1;
            struct ill *p = cg->spouses[node];
            while (p) {
                stack = ill_insert_front(stack, p->key, 0);
                p     = p->next;
            }
            p = cg->children[node];
            while (p) {
                stack = ill_insert_front(stack, p->key, 0);
                p     = p->next;
            }
        }
        NEXT: ;
    }
    CLEANUP: ;
    if (stack)
        ill_free(stack);
    free(marked);
    return unblocked_path;
}

int is_valid_insertion(struct cgraph *cg, struct gesrec g, int *onion,
                                          int onion_size)
{
    if (is_clique(cg, onion, onion_size)) {
        if (!unblocked_semidirected_path(cg, g.y, g.x, onion, onion_size))
            return 1;
    }
    return 0;
}

struct gesrec score_powerset(struct cgraph *cg, struct dataframe df,
                                                struct gesrec g, double *dscore,
                                                score_func score, double *fargs,
                                                int *iargs)
{
    double min_ds = 1.0f;
    struct gesrec min_g;
    min_g.x         = g.x;
    min_g.y         = g.y;
    min_g.set_size  = 0;
    min_g.naxy_size = g.naxy_size;
    min_g.set       = NULL;
    min_g.naxy      = malloc(min_g.naxy_size * sizeof(int));
    if (min_g.naxy == NULL)
        error("failed to allocate memory for naxy in fges!\n");
    memcpy(min_g.naxy, g.naxy, min_g.naxy_size * sizeof(int));

    // if (!is_clique(cg, min_g.naxy, min_g.naxy_size)) {
    //     Rprintf("got ya!\n");
    //     *dscore = min_ds;
    //     return min_g;
    // }

    /* saute in butter for best results */
    int *onion = malloc((g.naxy_size + g.set_size) * sizeof(int));
    if (onion == NULL)
        error("failed to allocate memory for onion in fges!\n");
    for (int i = 0; i < g.naxy_size; ++i)
        onion[i] = g.naxy[i];

    uint64_t n = 1 << g.set_size;
    for (int i = 0; i < n; ++i) {
        int onion_size = 0;
        for (uint32_t j = 0; j <  (uint32_t) g.set_size; ++j) {
            if ((i & (1 << j)) == (1 << j)) {
                onion[g.naxy_size + onion_size] = g.set[j];
                onion_size++;
            }
        }
        onion_size += g.naxy_size;
        if (DEBUG > 2) {
            for (int i = 0; i < onion_size; ++i)
                Rprintf("%i ", onion[i]);
            Rprintf("\n");
        }
        if (is_valid_insertion(cg, g, onion, onion_size)) {
            struct ill *parents = cg->parents[g.y];
            int         npar    = onion_size + ill_size(parents);
            int        *ypar    = malloc(npar * sizeof(int));
            if (ypar == NULL)
                error("failed to allocate memory for xy in fges!\n");
            for (int j = 0; j < onion_size; ++j)
                ypar[j] = onion[j];
            int j = onion_size;
            while (parents) {
                ypar[j] = parents->key;
                parents = parents->next;
                j++;
            }
            double ds = bic_score(df, g.x, g.y, ypar, npar, fargs, iargs);
            free(ypar);
            if (ds < min_ds) {
                min_ds = ds;
                min_g.set_size = onion_size - g.naxy_size;
                min_g.set = malloc(min_g.set_size * sizeof(int));
                for (int j = 0; j < min_g.set_size; ++j) {
                    min_g.set[j] = onion[j + min_g.naxy_size];
                }
            }
        }
    }
    free(onion);
    *dscore = min_ds;
    return min_g;
}



 void insert(struct cgraph *cg, struct gesrec g)
{
    if (DEBUG > 0 )
        Rprintf("insert %i -- > %i\n", g.x, g.y);
    add_edge_to_cgraph(cg, g.x, g.y, DIRECTED);
    for (int i = 0; i < g.set_size; ++i) {
        if (DEBUG > 0)
            Rprintf("orient %i -- > %i\n", g.set[i], g.y);
        orient_undirected_edge(cg, g.set[i], g.y);
    }
}

double recalcluate_node(struct dataframe df, struct cgraph *cg,
                                             struct gesrec *gesrecp,
                                             score_func score, double *fargs,
                                             int *iargs)
{
    double     dscore = 1.0f;
    double min_dscore = 1.0f;
    int y = gesrecp->y;
    for (int x = 0; x < df.nvar; ++x) {
        if ((x == y) || adjacent_in_cgraph(cg, x, y))
            continue;
        struct gesrec g = {0};
        g.x = x;
        g.y = y;
        struct ill *l = cg->spouses[y];
        while (l) {
            if (adjacent_in_cgraph(cg, x, l->key))
                g.naxy_size += 1;
            else
                g.set_size += 1;
            l = l->next;
        }
        g.naxy = malloc(g.naxy_size * sizeof(int));
        g.set  = malloc(g.set_size * sizeof(int));
        /* we have to reiterate through the list */
        l = cg->spouses[y];
        int j = 0;
        int k = 0;
        while (l) {
            int z = l->key;
            if (adjacent_in_cgraph(cg, x, z)) {
                g.naxy[j] = z;
                j++;
            }
            else {
                g.set[k]  = z;
                k++;
            }
            l = l->next;
        }
        struct gesrec min_g = score_powerset(cg, df, g, &dscore, score,
                                                 fargs, iargs);
        free(g.set);
        free(g.naxy);
        if (dscore < min_dscore) {
            min_dscore = dscore;
            free(gesrecp->set);
            free(gesrecp->naxy);
            *gesrecp = min_g;
        }
        else {
            free(min_g.set);
            free(min_g.naxy);
        }
    }
    return min_dscore;
}


 void delete(struct cgraph *cg, struct gesrec g)
{
    delete_edge_from_cgraph(cg, g.x, g.y, DIRECTED);
    for (int i = 0; i < g.set_size; ++i) {
        //
        // orient_undirected_edge(cg, g.x, g.set[i]);
        // orient_undirected_edge(cg, g.y, g.set[i]);
    }
}

struct cgraph *ccf_fges(struct dataframe df, score_func score,
                                        double *fargs, int *iargs)
{
    struct cgraph *cg          = create_cgraph(df.nvar);
    double         graph_score = 0.0f;
    double         dscore      = 0.0f;
    /*
    * We need to set up the priority queue so we know which edge to add
    * (and the other relevant information) at each stage of fges. Roughly how
    * this works is that each the highest scoring edge incident in each node is
    * recorded and then we find the highest scoring edge of all of those by
    * using the heap data structure we have
    */
    struct gesrec *gesrecords = calloc(df.nvar, sizeof(struct gesrec));
    struct heap   *heap       = create_heap(df.nvar, gesrecords,
                                                     sizeof(struct gesrec));
    double        *dscores    = heap->keys;
    void         **records    = heap->data;
    int           *indices    = heap->indices;
    for (int i = 0; i < df.nvar; ++i) {
        records[i] = gesrecords + i;
        indices[i] = i;
    }
    /* STEP 0: score x --> y */
    for (int j = 0; j < df.nvar; ++j) {
        double min     = DBL_MAX;
        int    arg_min = -1;
        for (int i = 0; i < j; ++i) {
            double ds = bic_score(df, i, j, NULL, 0, fargs, iargs);
            if (ds < min) {
                min     = ds;
                arg_min = i;
            }
        }
        dscores[j]      = min;
        gesrecords[j].x = arg_min;
        gesrecords[j].y = j;
    }
    build_heap(heap);


    /* FORWARD EQUIVALENCE SEARCH (FES) */
    struct gesrec *gesrecp;
    /* extract the smallest gesrec from the heap and check to see if positive */
    while ((gesrecp = extract_heap(heap, &dscore)) && dscore <= 0) {
        graph_score += dscore;
        insert(cg, *gesrecp);
        // ccf_chickering(cg = ccf_pdx(cg));
        //int *marked = calloc(df.nvar, sizeof(int));
        // int *nodes  = calloc(df.nvar, sizeof(int));
        int n_nodes = 0;
        int y = gesrecp->y;
        int x = gesrecp->x;
        int *visited = meek_local(cg, &y, 1, &n_nodes);
        if(!visited[x] && gesrecords[x].y == y) {
            visited[x] = 1;
            n_nodes++;
        }
        if(!visited[y]) {
            visited[y] = 1;
            n_nodes++;
        }
        for(int i = 0; i < gesrecp->set_size; ++i) {
            if(!visited[gesrecp->set[i]]) {
                visited[gesrecp->set[i]] = 1;
                n_nodes++;
            }
}
        int *nodes = calloc(n_nodes, sizeof(int));
        int j = 0;
        for(int i = 0; i < df.nvar; ++i) {
            if(visited[i]) {
                nodes[j] = i;
                j++;
            }
        }
        free(visited);
        // connected_nodes(gesrecp->y, marked, nodes, &n_nodes, cg);
        if (DEBUG) {
            print_cgraph(cg);
            for(int i = 0; i < n_nodes; ++i)
                Rprintf("%i ", nodes[i]);
            Rprintf("\n");
        }
        double dscore = recalcluate_node(df, cg, gesrecp, score, fargs,
                                             iargs);
        insert_heap(heap, dscore, gesrecp);
        for(int i = 0; i < n_nodes ; ++i) {
            if(nodes[i] == y)
                continue;
            struct gesrec *p = gesrecords + nodes[i];
            remove_heap(heap, nodes[i]);
            double dscore = recalcluate_node(df, cg, p, score, fargs, iargs);
            insert_heap(heap, dscore, p);

        }
        n_nodes = 0;
        free(nodes);
        if (DEBUG > 0) {
            for(int i = 0; i < df.nvar; ++i) {
                gesrecp = heap->data[i];
                Rprintf("%i --> %i, %f\n", gesrecp->x, gesrecp->y, heap->keys[i]);
            }
        }
        //free(marked);
    }


    /* BACKWARD EQUIVALENCE SEARCH (FES) */
    while (0) {
        /* TODO */
    }
    if (0) {
        delete(cg, *gesrecp);
    }

    for (int i = 0; i < df.nvar; ++i) {
        free(gesrecords[i].set);
        free(gesrecords[i].naxy);
    }
    free(gesrecords);
    //print_cgraph(cg);
    free_heap(heap);
    Rprintf("FES complete\n");
    return cg;
}
