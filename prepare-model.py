#!/usr/bin/env python


import os, sys


class CopiedFromPortal:

    def __init__(self, model):
        self.model = model
   
    def prepareModel(self):
        import tarfile
        from os.path import basename, dirname, splitext
        
        tgz = tarfile.open(self.model, 'r:gz')
        path = "model"

        directories = []
        fortranSourceFiles = []

        for tarinfo in tgz:
            if tarinfo.isdir():
                # Extract directory with a safe mode, so that
                # all files below can be extracted as well.
                try:
                    os.makedirs(os.path.join(path, tarinfo.name), 0777)
                except EnvironmentError:
                    pass
                directories.append(tarinfo)
            elif tarinfo.name.endswith(".f90"):
                pathname = os.path.join(path, tarinfo.name)
                fortranSourceFiles.append(pathname)
                thisDir = dirname(tarinfo.name) # see bcast_model.c
                s = tgz.extractfile(tarinfo)
                f = open(pathname, "w")
                # Preprocess.
                for line in s.readlines():
                    line = line.replace('@THIS_DIR@', thisDir)
                    f.write(line)
            else:
                tgz.extract(tarinfo, path)

        # Reverse sort directories.
        directories.sort(lambda a, b: cmp(a.name, b.name))
        directories.reverse()

        # Set correct owner, mtime and filemode on directories.
        for tarinfo in directories:
            path = os.path.join(path, tarinfo.name)
            try:
                tgz.chown(tarinfo, path)
                tgz.utime(tarinfo, path)
                tgz.chmod(tarinfo, path)
            except tarfile.ExtractError, e:
                pass

        # Generate the make include file.
        s = open("model.mk", "w")
        print >>s
        print >>s, "model_OBJECTS = \\"
        for sourceFile in fortranSourceFiles:
            base = splitext(basename(sourceFile))[0]
            print >>s, "\t$O/%s.o \\" % base
        print >>s, "\t$(empty)"
        print >>s
        for sourceFile in fortranSourceFiles:
            base = splitext(basename(sourceFile))[0]
            print >>s, "$O/%s.o: constants.h %s" % (base, sourceFile)
            print >>s, "\t${MPIFCCOMPILE_CHECK} -c -o $O/%s.o ${FCFLAGS_f90} %s" % (base, sourceFile)
            print >>s
        return


def prepareModel():
    import tarfile

    model = "model.tgz"

    modelDir = os.environ['MODEL']
    tgzOut = tarfile.open(model, 'w:gz')
    tgzOut.dereference = True # follow symlinks
    tgzOut.add(modelDir)
    tgzOut.close()

    CopiedFromPortal(model).prepareModel()


try:
    prepareModel()
except Exception, e:
    sys.exit("%s: %s" % (__file__, e))
