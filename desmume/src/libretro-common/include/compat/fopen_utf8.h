#ifndef __FOPEN_UTF8_H
#define __FOPEN_UTF8_H

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef _WIN32
void *fopen_utf8(const char * filename, const char * mode);
int stat_utf8(const char * filename, struct stat *buffer);
int access_utf8(const char * filename, int mode);
#else
#define fopen_utf8 fopen
#define stat_utf8 stat
#define access_utf8 access
#endif

#ifdef __cplusplus
}
#endif

#endif
