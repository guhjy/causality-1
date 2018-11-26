#include <stdlib.h>

#include "headers/causality.h"
#include "headers/cgraph.h"
#include "headers/int_linked_list.h"


struct cll {
    struct ill **children;
    struct ill **parents;
    struct ill **spouses;
    struct cll  *next;
};

/* a node is a sink if it has no children */
static int is_sink(struct cll *node)
{
     return *(node->children) == NULL;
}

/*
 * clique checks each undirected parent of current if that undirected
 * parent forms a clique with all the other parents of current
 */
static int is_clique(struct cll *node, struct cgraph *cg)
{
    struct ill *spouses = *(node->spouses);
    /* grab a spouse (undirected adjacent) */
    while (spouses) {
        int          spouse = spouses->key;
        struct ill *parents = *(node->parents);
        /* make sure spouse is adjacent to the parents of node */
        while (parents) {
            if (!adjacent_in_cgraph(cg, spouse, parents->key))
                return 0;
            parents = parents->next;
        }
        /* make sure spouse is adjacent to the other spouses of node */
        struct ill *p = *(node->spouses);
        while (p) {
            int spouse2 = p->key;
            if (spouse2 != spouse && !adjacent_in_cgraph(cg, spouse, spouse2))
                return 0;
            p = p->next;
        }
        spouses = spouses->next;
    }
    return 1;
}

static void orient_in_cgraph(struct cgraph *cg, int node)
{
    struct ill *cpy = copy_ill(cg->spouses[node]);
    struct ill *p   = cpy;
    while (p) {
        orient_undirected_edge(cg, p->key, node);
        p = p->next;
    }
    free(cpy);
}

static void remove_node(struct cll *current, struct cll *nodes)
{
    int node = current - nodes; /* ptr arithemtic */
    /* delete all listings of node in its parents and spouses */
    struct ill *parents = *(current->parents);
    while (parents) {
        ill_delete(nodes[parents->key].children, node);
        parents = parents->next;
    }
    struct ill *spouses = *(current->spouses);
    while (spouses) {
        ill_delete(nodes[spouses->key].spouses, node);
        spouses = spouses->next;
    }
}

struct cgraph * ccf_pdx(struct cgraph *cg)
{
    int            n_nodes = cg->n_nodes;
    struct cgraph *cpy     = copy_cgraph(cg);
    if (cpy == NULL) {
        free_cgraph(cg);
        CAUSALITY_ERROR("Failed to allocate memory for cpy in ccf_pdx\n");
        return NULL;
    }
    struct cll    *nodes   = calloc(n_nodes, sizeof(struct cll));
    if (nodes == NULL) {
        free_cgraph(cg);
        free_cgraph(cpy);
        CAUSALITY_ERROR("Failed to allocate memory for nodes in ccf_pdx\n");
        return NULL;
    }
    /* set up circular linked list */
    struct ill **parents  = cg->parents;
    struct ill **spouses  = cg->spouses;
    struct ill **children = cg->children;
    for (int i = 0; i < n_nodes; ++i) {
        nodes[i].parents  = parents  + i;
        nodes[i].children = children + i;
        nodes[i].spouses  = spouses  + i;
        nodes[i].next     = nodes + (i + 1) % n_nodes;
    }
    struct cll *current   = nodes;
    struct cll *prev      = nodes + (n_nodes - 1);
    int         n_checked = 0;
    int         ll_size   = n_nodes;
    /* Comment needed */
    while (ll_size > 0 && n_checked <= ll_size) {
        if (is_sink(current) && is_clique(current, cg)) {
            orient_in_cgraph(cpy, current - nodes);
            remove_node(current, nodes);
            prev->next = current->next;
            ll_size--;
            n_checked = 0;
        }
        else {
            n_checked++;
            prev = prev->next;
        }
        current = current->next;
    }
    free_cgraph(cg);
    free(nodes);
    /*
     * check to see if pdx failed to generate an extension. If there is a
     * failure, free the copy_ptr and set it to NULL.
     */
    int failure = ll_size > 0 ? 1 : 0;
    if (failure) {
        free_cgraph(cpy);
        cpy = NULL;
    }
    return cpy;
}
