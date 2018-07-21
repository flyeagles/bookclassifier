import argparse
import os
import stat

class FolderBrowser:
    def __init__(self, root):
        self.root = root
    
    def start(self):
        self.recursive(self.root)

    def visit_file(self, pathname):
        # child class should overwrite this
        pass
    
    def visit_folder(self, folder):
        # child class should overwrite this
        pass


    def recursive(self, rootdir):
        for diritem in os.listdir(rootdir):
            pathname = os.path.join(rootdir, diritem)

            try:
                statdata = os.stat(pathname)
            except FileNotFoundError as file_err:
                print(file_err)
                print("Skip file " + pathname)
                continue

            mode = statdata.st_mode
            if stat.S_ISDIR(mode):
                # It's a directory, recurse into it
                self.visit_folder(pathname)
                self.recursive(pathname)
            elif stat.S_ISREG(mode):
                self.visit_file(pathname)
            else:
                # Unknown file type, print a message
                print('Skipping %s' % pathname)


class FolderCopyBrowser(FolderBrowser):
    def __init__(self, source, target):
        self.source = source
        self.sourcelen = len(source)
        self.target = target
        self.first = True

        super().__init__(source)
    
    def visit_file(self, pathname):
        # ignore files for folder copy
        pass

    def visit_folder(self, folder):
        print("---", folder)
        if self.first:
            self.first = False
            os.mkdir(self.target)
        os.mkdir(self.target + '/' +  folder[self.sourcelen:])
        #os.mkdir(os.path.join(self.target, folder[self.sourcelen:]))




if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Copy only folder structures')
    argparser.add_argument("-s", dest='source_folder', # metavar='Folder-root',
                           type=str, default=None, required=True,
                           help='root of the source folder tree')
    argparser.add_argument("-t", dest='target_folder', # metavar='Folder-root',
                           type=str, default=None, required=True,
                           help='root of the target folder tree')
    '''
    argparser.add_argument("-t", dest='target', # metavar='Folder-root',
                           default=False, required=False, action='store_true',
                           help='work in training moode')

    argparser.add_argument("--work", dest='work', # metavar='Folder-root',
                           type=str, default=None, required=False, 
                           help='work in work moode, specify the work folder.')
    '''

    args = argparser.parse_args()

    print(args)

    try:
        statdata = os.stat(args.source_folder)
    except FileNotFoundError as file_err:
        print(file_err)
        print("Source folder must exist. but '{s}' is missing.".format(s=args.source_folder))
        exit(1)

    try:
        statdata = os.stat(args.target_folder)
        print("Target folder must not exist. but '{s}' does exist.".format(s=args.target_folder))
        exit(1)
    except FileNotFoundError as file_err:
        pass


    foldercopier = FolderCopyBrowser(args.source_folder, args.target_folder)
    foldercopier.start()
