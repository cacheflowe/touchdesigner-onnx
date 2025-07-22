import sys
import os
import platform
import td

def PrintPythonPath():
	print("[Config] 🐍----------------------------------🐍")
	print('[Config] Python sys.path:')
	for path in sys.path:
		print("[Config] -", path)
	print("[Config] 🐍----------------------------------🐍")


def AddCondaEnvToPath(user, env_name):
	if platform.system() == 'Windows':
		windowsPathBase = 'C:/Users/'+user+'/miniconda3/envs/'+env_name
		windowsPathDLLs = windowsPathBase+'/DLLs'
		windowsPathLib = windowsPathBase+'/Library/bin'
		windowsPathSite = windowsPathBase+'/Lib/site-packages'
		if windowsPathSite not in sys.path:
			print(f"[Config] Adding conda environment '{env_name}' for user '{user}' to sys.path")
			print('[Config] Added Conda DLLs and Library paths added to sys.path:')
			print('[Config] - Conda env DLLs path: ', windowsPathDLLs)
			print('[Config] - Conda env Library path: ', windowsPathLib)
			print('[Config] - Conda env site-packages path: ', windowsPathSite)
			os.add_dll_directory(windowsPathDLLs)
			os.add_dll_directory(windowsPathLib)
			sys.path.insert(0, windowsPathSite)  # Add to the beginning of the path list
		else:
			print('[Config] Conda env {} already loaded!'.format(env_name))
	else:
		print(f"[Config] Adding conda environment '{env_name}' for user '{user}' to sys.path")
		macPathBase = '/Users/'+user+'/opt/miniconda3/envs/'+env_name
		macPathLib = macPathBase+'/lib'
		macPathBin = macPathBase+'/bin'
		macPathSite = macPathBase+'/lib/python3.9/site-packages'
		if macPathSite not in sys.path:
			print('[Config] Added Conda lib, bin and site-packages paths to sys.path:')
			print('[Config] - Conda env lib path: ', macPathLib)
			print('[Config] - Conda env bin path: ', macPathBin)
			print('[Config] - Conda env site-packages path: ', macPathSite)
			os.environ['PATH'] = macPathLib + os.pathsep + os.environ['PATH']
			os.environ['PATH'] = macPathBin + os.pathsep + os.environ['PATH']
			sys.path.insert(0, macPathSite)  # Add to the beginning of the path list
		else:
			print('[Config] Conda env {} already loaded!'.format(env_name))


def AddPyDirToPath(new_path):
	if new_path not in sys.path:  # Check if the path is already in the list
		if os.path.exists(new_path):  # Check if the path exists on disk
			sys.path.insert(0, new_path)  # Add to the beginning of the path list
	else:
		print('[Config] Python path already loaded!')


