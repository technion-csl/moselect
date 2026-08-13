
#ifndef _MEM_ALLOC_HOOK_H_
#define _MEM_ALLOC_HOOK_H_

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstddef>

/* MORECORE should be defined before including dlmalloc header file */
#define MORECORE mosalloc_morecore
#include "malloc.h"

#ifdef __cplusplus
 extern "C" {
#endif /* __cplusplus */

int mprotect(void *addr, size_t len, int prot) __THROW;
void *mmap(void *addr, size_t length, int prot, int flags, int fd, 
           off_t offset) __THROW;
int munmap(void *addr, size_t length) __THROW;

int brk(void *addr) __THROW;
void *sbrk(ptrdiff_t increment) __THROW;
void *mosalloc_morecore(ptrdiff_t increment) __THROW;

#ifdef __cplusplus
}  /* end of extern "C" */
#endif /* __cplusplus */

#endif //_MEM_ALLOC_HOOK_H_
