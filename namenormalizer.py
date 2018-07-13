import re
import stat
import os
import argparse


def browse_folder(rootdir, change):
    old_cwd = os.getcwd()
    os.chdir(rootdir)

    change_count = 0
    for diritem in os.listdir('.'):
        pathname = diritem  #os.path.join(rootdir, diritem)

        try:
            statdata = os.stat(pathname)
        except FileNotFoundError as file_err:
            print(file_err)
            print("Skip file " + pathname)
            continue

        mode = statdata.st_mode
        if stat.S_ISDIR(mode):
            print('====', diritem)
            # It's a directory, recurse into it
            change_count += browse_folder(diritem, change)
        elif stat.S_ISREG(mode):
            # It's a file, call the callback function
            newfilename = diritem
            newfilename = re.sub('扫描版', '', newfilename)
            newfilename = re.sub('［', '[', newfilename)
            newfilename = re.sub('【', '[', newfilename)
            newfilename = re.sub('］', ']', newfilename)
            newfilename = re.sub('】', ']', newfilename)
            newfilename = re.sub('）', ')', newfilename)
            newfilename = re.sub('（', '(', newfilename)
            newfilename = re.sub('\s+', '.', newfilename)
            newfilename = re.sub('\.+', '.', newfilename)
            # newfilename = re.sub('\]\.', ']', newfilename)  this might cause name like "2012]pdf"
            newfilename = re.sub('\.\[', '[', newfilename)

            #newfilename = re.sub('·', '.', newfilename)

            newfilename = re.sub(r'^\[', '', newfilename)

            newfilename = re.sub(r'\]pdf', '].pdf', newfilename)


            if newfilename != diritem:
                # need rename file
                change_count += 1
                print(diritem, '---->', newfilename)
                if change:
                    os.rename(diritem, newfilename)
        else:
            # Unknown file type, print a message
            print('Skipping %s' % pathname)

    os.chdir(old_cwd)
    return change_count

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Normalized book file names.')
    #argparser.add_argument("--RAM", dest='RAM_disk', # metavar='Folder-root',
    #                       type=str, default=None, required=True,
    #                       help='root of the RAM disk')
    argparser.add_argument("--change", dest='change', # metavar='Folder-root',
                           default=False, required=False, action='store_true',
                           help='Do file name changes.')

    args = argparser.parse_args()

    change_count = browse_folder('.', args.change)
    print("Change {cnt} files.".format(cnt=change_count))


