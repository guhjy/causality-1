#include"headers/rapi.h"

#include<stdlib.h>
#include<string.h>
#include <stdio.h>

#define IS_RED(node) ((node) == NULL ? 0 : ((node)->key > 0 ? 1: 0))
#define ABS(x) ((x) > 0 ? (x): -(x))

#define SET_TO_RED(node) ((node)->key = ABS((node)->key))
#define SET_TO_BLACK(node) ((node)->key = -ABS((node)->key))

#define LEFT 0
#define RIGHT 1

typedef struct int_rbt_node* int_rbt_node_ptr;
typedef struct int_rbt_node {
  int key;
  int_rbt_node_ptr child[2];
  int values [];
}int_rbt_node;

int int_rbt_key(int_rbt_node_ptr root) {
  return(ABS(root->key) - 1);
}

inline int_rbt_node_ptr int_rbt_instantiate_node(const int key, const int n,
                                          const int * const values) {
  int_rbt_node_ptr tmp = malloc(sizeof(int_rbt_node) + n*sizeof(int));
  if(tmp == NULL) {
    error("failed to allocate memory for rbt pointer\n");
  }

  memcpy(tmp->values, values, n*sizeof(int));
  tmp->key          = key + 1;
  tmp->child[LEFT]  = NULL;
  tmp->child[RIGHT] = NULL;
  return(tmp);
}

inline int_rbt_node_ptr single_rotation(int_rbt_node_ptr root, int direction) {
  int_rbt_node_ptr tmp = root->child[!direction];

  root->child[!direction] = tmp->child[direction];
  tmp->child[direction]   = root;

  SET_TO_RED(root);
  SET_TO_BLACK(tmp);
  return tmp;
}

inline int_rbt_node_ptr double_rotation(int_rbt_node_ptr root, int direction) {

  root->child[!direction] = single_rotation(root->child[!direction],
                                            !direction);
  return single_rotation(root, direction);
}

int_rbt_node_ptr int_rbt_insert_recurvise(int_rbt_node_ptr root,
                                                 const int key,
                                                 const int n,
                                                 const int* const values)
{
  if(root == NULL) {
    root = int_rbt_instantiate_node(key, n, values);
  }
  else if (key == int_rbt_key(root)) {
    for(int i = 0; i < n; ++i)
      root->values[i] += values[i];
  }
  else {
    int direction = key < int_rbt_key(root) ? LEFT : RIGHT;
    root->child[direction] = int_rbt_insert_recurvise(root->child[direction],
                                                      key, n, values);

    if(IS_RED(root->child[direction])) {

      if(IS_RED(root->child[!direction])) {

        SET_TO_RED(root);
        SET_TO_BLACK(root->child[LEFT]);
        SET_TO_BLACK(root->child[RIGHT]);
      }
      else {
        if(IS_RED(root->child[direction]->child[direction]))
          root = single_rotation(root, !direction);
        else if(IS_RED(root->child[direction]->child[!direction]))
          root = double_rotation(root, !direction);
      }
    }
  }
  return root;
}

int_rbt_node_ptr int_rbt_insert(int_rbt_node_ptr root, const int key, const int n,
                                const int * const values)
{
  int_rbt_node_ptr tmp = int_rbt_insert_recurvise(root, key ,n, values);
  SET_TO_BLACK(tmp);
  return tmp;
}

void int_rbt_print_tree(int_rbt_node_ptr root, const int n) {
  if(root != NULL) {
    Rprintf("Key: %i Value(s):", int_rbt_key(root));
    for(int i = 0; i < n; ++i)
      Rprintf(" %i" , root->values[i]);
    Rprintf("\n");
    int_rbt_print_tree(root->child[LEFT], n);
    int_rbt_print_tree(root->child[RIGHT], n);
  }
}

void int_rbt_free(int_rbt_node_ptr root) {
  if(root != NULL) {
    int_rbt_free(root->child[LEFT]);
    int_rbt_free(root->child[RIGHT]);
    free(root);
  }
}

int_rbt_node_ptr int_rbt_merge_trees(int_rbt_node_ptr dst, int_rbt_node_ptr src,
                          const int n)
{
  if(src != NULL) {
    dst = int_rbt_merge_trees(dst, src->child[LEFT], n);
    dst = int_rbt_merge_trees(dst, src->child[RIGHT], n);
    return(int_rbt_insert(dst, int_rbt_key(src), n , src->values));
  }
  else
    return dst;
}

int int_rbt_size(int_rbt_node_ptr root) {
  if(root != NULL)
    return(1 + int_rbt_size(root->child[LEFT]) +
            int_rbt_size(root->child[RIGHT]));
  else
    return 0;
}

int* int_rbt_values_ptr(int_rbt_node_ptr root) {
  return (root->values);
}

int_rbt_node_ptr int_rbt_left_child(int_rbt_node_ptr root) {
  return (root->child[LEFT]);
}

int_rbt_node_ptr int_rbt_right_child(int_rbt_node_ptr root) {
  return (root->child[RIGHT]);
}