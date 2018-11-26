#include <cgraph.h>
#include <stdio.h>

#include "ges.h"

#ifndef CAUSALITY_H
#define CAUSALITY_H

#define CAUSALITY_ERROR(s) fprintf(stderr, "%s\n", s);

#define DIRECTED      1 /* -->               */
#define UNDIRECTED    2 /* ---               */
#define PLUSPLUSARROW 3 /* ++> aka --> dd nl */
#define SQUIGGLEARROW 4 /* ~~> aka --> pd nl */
#define CIRCLEARROW   5 /* o->               */
#define CIRCLECIRCLE  6 /* o-o               */
#define BIDIRECTED    7 /* <->               */

#define NUM_NL_EDGETYPES 2
#define NUM_LAT_EDGETYPES 7
#define NUM_EDGES_STORED 11

#define IS_DIRECTED(edge) (edge == DIRECTED || edge == CIRCLEARROW || \
                           edge == SQUIGGLEARROW || edge == PLUSPLUSARROW)


/* Search algorithms */
struct cgraph * ccf_ges(struct ges_score score);
/* Graph manipulations */
int           * ccf_sort(struct cgraph *cg);
struct cgraph * ccf_pdx(struct cgraph *cg);
void            ccf_chickering(struct cgraph *cg);

double ccf_score_graph(struct cgraph *cg, struct dataframe df, score_func score,
                                      struct score_args args);
#endif
