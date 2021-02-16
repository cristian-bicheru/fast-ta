import subprocess
import sys
import os
from sysconfig import get_paths

# Attempt to see if running python installation has everything we need.
found_include = False
found_numpy_include = False

if sys.version_info[0] == 3:
    print(get_paths()['include'])
    found_include = True
    try:
        import numpy
        print(numpy.get_include())
        found_numpy_include = True
    except ImportError:
        pass 

if not found_include:
    try:
        subprocess.check_output(["python3", "--version"]) # ensure error is raised if it does not exist
        print(subprocess.check_output("python3 -c'from sysconfig import get_paths;print(get_paths()[\"include\"])'", shell=True).strip().decode())
        found_include = True
        if not found_numpy_include:
            try:
                print(subprocess.check_output('python3 -c"import numpy;print(numpy.get_include())"', shell=True).strip().decode())
                found_numpy_include = True
            except:
                pass
    except:
        pass

if not found_include:
    for i in range(1, 10):
        try:
            subprocess.check_output(["python3.%i"%i, "--version"]) # ensure error is raised if it does not exist
            print(subprocess.check_output("python3.%i -c'from sysconfig import get_paths;print(get_paths()[\"include\"])'"%i, shell=True).strip().decode())
            found_include = True
            if not found_numpy_include:
                try:
                    print(subprocess.check_output('python3.%i -c"import numpy;print(numpy.get_include())"'%i, shell=True).strip().decode())
                    found_numpy_include = True
                except:
                    pass
            break
        except:
            pass
