#include <compat/fopen_utf8.h>
#include <encodings/utf.h>
#include <stdlib.h>

#if defined(_WIN32_WINNT) && _WIN32_WINNT < 0x0500 || defined(_XBOX)
#ifndef LEGACY_WIN32
#define LEGACY_WIN32
#endif
#endif

#ifdef _WIN32
#undef fopen

FILE* fopen_utf8(const char * filename, const char * mode)
{
#if defined(_XBOX)
   return fopen(filename, mode);
#elif defined(LEGACY_WIN32)
   FILE             *ret = NULL;
   char * filename_local = utf8_to_local_string_alloc(filename);

   if (!filename_local)
      return NULL;
   ret = fopen(filename_local, mode);
   if (filename_local)
      free(filename_local);
   return ret;
#else
   wchar_t * filename_w = utf8_to_utf16_string_alloc(filename);
   wchar_t * mode_w = utf8_to_utf16_string_alloc(mode);
   FILE* ret = _wfopen(filename_w, mode_w);
   free(filename_w);
   free(mode_w);
   return ret;
#endif
}

#undef stat
int stat_utf8(const char * filename, struct stat *buffer)
{
    wchar_t * filename_w = utf8_to_utf16_string_alloc(filename);
    struct _stat params;

    int ret = _wstat(filename_w, &params);
    free(filename_w);

    if (buffer && !ret)
    {
        buffer->st_gid   = params.st_gid;
        buffer->st_atime = params.st_atime;
        buffer->st_ctime = params.st_ctime;
        buffer->st_dev   = params.st_dev;
        buffer->st_ino   = params.st_ino;
        buffer->st_mode  = params.st_mode;
        buffer->st_mtime = params.st_mtime;
        buffer->st_nlink = params.st_nlink;
        buffer->st_rdev  = params.st_rdev;
        buffer->st_size  = params.st_size;
        buffer->st_uid   = params.st_uid;
    }

    return ret;
}

#undef access
int access_utf8(const char * filename, int mode)
{
    wchar_t * filename_w = utf8_to_utf16_string_alloc(filename);
    int ret = _waccess(filename_w, mode);
    free(filename_w);
    return ret;
}
#endif
