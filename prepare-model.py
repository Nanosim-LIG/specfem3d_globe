#!/usr/bin/env python


import os, sys


class MovedFromPortal:

    def __init__(self, model):
        self.model = model
   
    def prepareModel(self):
        import tarfile
        from os.path import basename, dirname, exists, join, splitext
        from itertools import chain
        
        tgz = tarfile.open(self.model, 'r:gz')
        root = "model"
        cwd = os.getcwd()

        directories = []
        serialFortranSourceFiles = []
        serialCSourceFiles = []
        fortranSourceFiles = []
        cSourceFiles = []
        bcastFiles = []

        for tarinfo in tgz:
            if tarinfo.isdir():
                # Extract directory with a safe mode, so that
                # all files below can be extracted as well.
                try:
                    os.makedirs(join(root, tarinfo.name), 0777)
                except EnvironmentError:
                    pass
                directories.append(tarinfo)
            else:
                tgz.extract(tarinfo, root)
                if not "/shared/" in tarinfo.name:
                    bcastFiles.append(tarinfo.name)
            
            if tarinfo.name.endswith(".f90") or tarinfo.name.endswith(".c"):
                pathname = join(root, tarinfo.name)
                os.unlink(pathname)
                if tarinfo.name.endswith(".f90"):
                    if tarinfo.name.endswith(".serial.f90"):
                        serialFortranSourceFiles.append(pathname)
                    else:
                        fortranSourceFiles.append(pathname)
                else:
                    if tarinfo.name.endswith(".serial.c"):
                        serialCSourceFiles.append(pathname)
                    else:
                        cSourceFiles.append(pathname)
                thisDir = dirname(tarinfo.name) # see bcast_model.c
                s = tgz.extractfile(tarinfo)
                f = open(pathname, "w")
                # Preprocess.
                for line in s.readlines():
                    line = line.replace('@THIS_DIR@', thisDir)
                    f.write(line)

        # Create symlinks to "shared" directories.
        sharedParents = []
        for directory, subdirectories, files in os.walk(root):
            for subdirectory in subdirectories:
                if subdirectory == "shared":
                    pathname = join(directory, "shared")
                    symLink = join(directory, "_shared")
                    if exists(symLink):
                        os.unlink(symLink)
                    os.symlink(join(cwd, pathname), symLink)
                    sharedParents.append(directory[len(root)+1:])

        # Reverse sort directories.
        directories.sort(lambda a, b: cmp(a.name, b.name))
        directories.reverse()

        # Set correct owner, mtime and filemode on directories.
        for tarinfo in directories:
            pathname = os.path.join(root, tarinfo.name)
            try:
                tgz.chown(tarinfo, pathname)
                tgz.utime(tarinfo, pathname)
                tgz.chmod(tarinfo, pathname)
            except tarfile.ExtractError, e:
                pass

        # Generate the bcast tgz file.
        tgz = tarfile.open("bcast_model.tgz", 'w:gz')
        for name in bcastFiles:
            tgz.add(join(root, name), name)
        for name in sharedParents:
            tgz.add(join(root, name, "_shared"), join(name, "shared"))
        tgz.close()

        # Generate the make include file.
        s = open("model.mk", "w")
        print >>s
        print >>s, "model_SERIAL_OBJECTS = \\"
        for sourceFile in chain(serialFortranSourceFiles, serialCSourceFiles):
            base = splitext(basename(sourceFile))[0]
            print >>s, "\t$O/%s.o \\" % base
        print >>s, "\t$(empty)"
        print >>s
        print >>s, "model_OBJECTS = \\"
        for sourceFile in chain(fortranSourceFiles, cSourceFiles):
            base = splitext(basename(sourceFile))[0]
            print >>s, "\t$O/%s.o \\" % base
        print >>s, "\t$(model_SERIAL_OBJECTS) \\"
        print >>s, "\t$(empty)"
        print >>s
        for sourceFile in serialFortranSourceFiles:
            base = splitext(basename(sourceFile))[0]
            print >>s, "$O/%s.o: constants.h %s" % (base, sourceFile)
            print >>s, "\t${FCCOMPILE_CHECK} -c -o $O/%s.o ${FCFLAGS_f90} %s" % (base, sourceFile)
            print >>s
        for sourceFile in serialCSourceFiles:
            base = splitext(basename(sourceFile))[0]
            print >>s, "$O/%s.o: config.h %s" % (base, sourceFile)
            print >>s, "\t$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $O/%s.o %s" % (base, sourceFile)
            print >>s
        for sourceFile in fortranSourceFiles:
            base = splitext(basename(sourceFile))[0]
            print >>s, "$O/%s.o: constants.h %s" % (base, sourceFile)
            print >>s, "\t${MPIFCCOMPILE_CHECK} -c -o $O/%s.o ${FCFLAGS_f90} %s" % (base, sourceFile)
            print >>s
        for sourceFile in cSourceFiles:
            base = splitext(basename(sourceFile))[0]
            print >>s, "$O/%s.o: config.h %s" % (base, sourceFile)
            print >>s, "\tmpicc $(CPPFLAGS) $(CFLAGS) -c -o $O/%s.o %s" % (base, sourceFile)
            print >>s
        return


def prepareModel():
    import tarfile
    from os.path import isdir

    model = os.environ.get('MODEL')
    assert model, "MODEL environment variable is not set"
    
    if isdir(model):
        modelDir = model
        model = "model.tgz"
        tgzOut = tarfile.open(model, 'w:gz')
        tgzOut.dereference = True # follow symlinks
        tgzOut.add(modelDir)
        tgzOut.close()

    MovedFromPortal(model).prepareModel()


prepareModel()
