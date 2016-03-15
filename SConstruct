
import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults
from SCons.Script import *

from subprocess import Popen, PIPE


# create a SCons environment
env = Environment()

def TOOL_CUDA(env):
    """SCons.Tool.nvcc

    Tool-specific initialization for NVIDIA CUDA Compiler.

    There normally shouldn't be any need to import this module directly.
    It will usually be imported through the generic SCons.Tool.Tool()
    selection method.

    """
    CUDASuffixes = ['.cu']

    def add_common_nvcc_variables(env):
      """
      Add underlying common "NVIDIA CUDA compiler" variables that
      are used by multiple builders.
      """
      print "Adding common nvcc variables"

      # "NVCC common command line"
      if not env.has_key('_NVCCCOMCOM'):
        # prepend -Xcompiler before each flag

        # these flags are common to both static and shared compilations
        env['_NVCCCOMCOM'] = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__)} $_CPPDEFFLAGS $_CPPINCFLAGS'

        # wrap up all these environment variables inside -Xcompiler ""
        env['_NVCCWRAPCFLAGS'] =     '${_concat("-Xcompiler ", CFLAGS,     "", __env__)}'
        env['_NVCCWRAPSHCFLAGS'] =   '${_concat("-Xcompiler ", SHCFLAGS,   "", __env__)}'
        env['_NVCCWRAPCCFLAGS'] =    '${_concat("-Xcompiler ", CCFLAGS,    "", __env__)}'
        env['_NVCCWRAPSHCCFLAGS'] =  '${_concat("-Xcompiler ", SHCCFLAGS,  "", __env__)}'
        # XXX should these be wrapped as well?  not sure -jph
        #env['_NVCCWRAPCXXFLAGS'] =   '${_concat("-Xcompiler ", CXXFLAGS,   "", __env__)}'
        #env['_NVCCWRAPSHCXXFLAGS'] = '${_concat("-Xcompiler ", SHCXXFLAGS, "", __env__)}'

    def generate(env):
      """
      Add Builders and construction variables for CUDA compilers to an Environment.
      """
      static_obj, shared_obj = SCons.Tool.createObjBuilders(env)
      print "in Generate"
      for suffix in CUDASuffixes:
        # Add this suffix to the list of things buildable by Object
        static_obj.add_action('$CUDAFILESUFFIX', '$NVCCCOM')
        shared_obj.add_action('$CUDAFILESUFFIX', '$SHNVCCCOM')
        static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
        shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)

        # Add this suffix to the list of things scannable
        SCons.Tool.SourceFileScanner.add_scanner(suffix, CUDAScanner)

      add_common_nvcc_variables(env)

      # set the "CUDA Compiler Command" environment variable
      env['NVCC'] = 'nvcc'
      env['SHNVCC'] = 'nvcc'
      
      # set the include path, and pass both c compiler flags and c++ compiler flags
      env['NVCCFLAGS'] = SCons.Util.CLVar('')
      env.Append(NVCCFLAGS = "--std=c++11")
      #env.Append(NVCCFLAGS = ['-gencode=arch=compute_30,code=sm_30'])
      env['SHNVCCFLAGS'] = SCons.Util.CLVar('') + ' -shared'

      env['NVCCCPPPATH'] = ['#']
      env['NVCCINCPREFIX'] = '-I'
      env['NVCCINCSUFFIX'] = ''
      env['_NVCCINCFLAGS'] = '${_concat(NVCCINCPREFIX, NVCCCPPPATH, NVCCINCSUFFIX, __env__, RDirs, TARGET, SOURCE)}'

      # 'NVCC Command'
      env['NVCCCOM']   = '$NVCC -o $TARGET -c $NVCCFLAGS $_NVCCINCFLAGS $_NVCCWRAPCFLAGS $_NVCCWRAPCCFLAGS $_NVCCCOMCOM $SOURCES'
      env['SHNVCCCOM'] = '$SHNVCC -o $TARGET -c $SHNVCCFLAGS $_NVCCWRAPSHCFLAGS $_NVCCWRAPSHCCFLAGS $_NVCCCOMCOM $SOURCES'
      
      # the suffix of CUDA source files is '.cu'
      env['CUDAFILESUFFIX'] = '.cu'

      # XXX add code to generate builders for other miscellaneous
      # CUDA files here, such as .gpu, etc.

      # XXX intelligently detect location of nvcc here
      exe_path = '/usr/local/cuda/bin'

      # XXX intelligently detect location of cuda libraries here
      lib_path = '/usr/local/cuda/lib64'
      
      env.PrependENVPath('PATH', exe_path)

    def exists(env):
      return env.Detect('nvcc')
    
    class CUDAScanner(SCons.Scanner.ClassicCPP):
        """
        Use a variant of the usual C scanner to find header files in .cu files
        Need this variant because nvcc looks in the compiler's invocation dir
        as well as source file's dir, which ClassicCPP doesn't handle.
        """
        def find_include(self, include, source_dir, path):
            if include[0] == '"':
            # Prepend top dir here for nvcc
                paths = (Dir('#'),) + (source_dir,) + tuple(path)
            else:
                paths = tuple(path) + (source_dir,)

            n = SCons.Node.FS.find_file(include[1], paths)
            #print "Finding include file %s in %s: %s"%(include, map(str, paths), n)
            i = SCons.Util.silent_intern(include[1])
            return n, i
    generate(env)

    CScanner = CUDAScanner("CUDAScanner",
                               env['CPPSUFFIXES'] + ['.cu'],
                               "NVCCCPPPATH",
                               '^[ \t]*#[ \t]*(?:include|import)[ \t]*(<|")([^>"]+)(>|")')
    SourceFileScanner.add_scanner('.cu', CScanner)


def TOOL_HIP(env):

    # here will be branch to generate PTX or CGN code
    # For now it is just a PTX version but the HCC version
    # will be created in the same way
#    if config.hipplatform = 'nvcc':
#        TOOL_HIP_NVCC(env)
#    elif config.hipplatform = 'hcc':
#        TOOL_HIP_HCC(env)
#    else:
#        raise Error, "Invalid platform";
        
    #Code for TOOL_HIP_NVCC

    hip_suffixes=['.hip.cpp']
    
    # Here we should put some code to find the hip. I have it visible in whole env
    result = Popen(['hipconfig','--path'], stdout=PIPE).communicate()[0]

    hip_bin = result + '/bin'
    hip_include = result + '/include'
    hipcc= result + '/bin/hipcc'

    env['HIPCC'] = 'hipcc'
    env['HIPCCCOM'] = '$HIPCC $HIP_ARCHFLAGS $HIPFLAGS $_HIPINCFLAGS $_HIPDEFFLAGS $HIP_DEBUGOPT $HIP_EXTRAFLAGS -c $SOURCES -o $TARGET'

    #Now we have to define all of above $ params
    env['HIPDEFINES'] = '__HIP_PLATFORM_NVCC__'
    env['HIPFLAGS'] = ["--std=c++11"]

    env['HIPDEFPREFIX'] = '-D'
    env['HIPDEFSUFFIX'] = ''
    env['_HIPDEFFLAGS'] = '${_defines(HIPDEFPREFIX, HIPDEFINES + CPPDEFINES, HIPSUFFIX, __env__)}'

    env['HIPCPPPATH'] = ['#']
    env['HIPINCPREFIX'] = '-I'
    env['HIPINCSUFFIX'] = ''
    env['_HIPINCFLAGS'] = '${_concat(HIPINCPREFIX, HIPCPPPATH, HIPINCSUFFIX, __env__, RDirs, TARGET, SOURCE)}'

    env['HIP_DEBUGOPT'] = SCons.Util.CLVar()
    env['HIP_EXTRAFLAGS'] = SCons.Util.CLVar()

    #libs should be added automatically based on hip configuration
    env.PrependUnique(HIPCPPPATH = hip_include)
    env.Append(HIPDEFINES ='__HIP_PLATFORM_NVCC__');
    env.PrependENVPath('PATH', hip_bin)
    
    #env.PrependUnique(LIBPATH=env.cuda_libdir)
    #env.PrependENVPath('LD_LIBRARY_PATH', env.cuda_libdir)
    #env.AppendUnique(CUDAFLAGS=['-Xcompiler', '-fPIC'])

    #if config.final and not config.profile
        #HIPFLAGS = -gencode=arch=compute_20,code=sm_20  etc...


    #Add custom hip builder
    action='$HIPCCCOM'
    hip_suffix = '.hip.cpp' #test this
    hipbuilder = Builder(action=action, src_suffix=hip_suffix, suffix='$OBJSUFFIX', prefix='$OBJPREFIX')
    env['BUILDERS']['HipObj'] = hipbuilder
    
    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)
    static_obj.add_action(hip_suffix, action)
    shared_obj.add_action(hip_suffix, action)
    shared_obj.add_emitter(hip_suffix, SCons.Defaults.SharedObjectEmitter)

    class HIPScanner(SCons.Scanner.ClassicCPP):
        """
        Use a variant of the usual C scanner to find header files in .cu files
        Need this variant because nvcc looks in the compiler's invocation dir
        as well as source file's dir, which ClassicCPP doesn't handle.
        """
        def find_include(self, include, source_dir, path):
            if include[0] == '"':
            # Prepend top dir here for nvcc
                paths = (Dir('#'),) + (source_dir,) + tuple(path)
            else:
                paths = tuple(path) + (source_dir,)
    
            n = SCons.Node.FS.find_file(include[1], paths)
            #print "Finding include file %s in %s: %s"%(include, map(str, paths), n)
            i = SCons.Util.silent_intern(include[1])
            return n, i

    CScanner = HIPScanner("HIPScanner", env['CPPSUFFIXES'] + ['.hip.cpp'], 
                           "HIPCPPPATH",
                           '^[ \t]*#[ \t]*(?:include|import)[ \t]*(<|")([^>"]+)(>|")')
    
    SourceFileScanner.add_scanner(hip_suffix, CScanner)



#Main SCons script code
os.environ['HIP_PLATFORM'] = 'nvcc'
#Update env with our custom builders
print "About to run TOOL CUDA"
TOOL_CUDA(env)
#print "About to run HIP TOOL"
#TOOL_HIP(env)

# Add the CUDA library paths to the LIBPATH
env.Append(LIBPATH  = ['/usr/local/cuda/lib64'])
env.PrependUnique(CPPPATH= '/usr/local/cuda/include/')
env.PrependUnique(NVCCCPPPATH='/usr/local/cuda/include/')


# Link to cuda runtime and c++ runtime
# 'stdc++' is for c++, pthread for c++ 11
env.Append(LIBS = ['cudart','stdc++', 'pthread', 'm', 'X11'])

# set flags for release/debug/emurelease/emudebug
flags = ['-O3']
nvflags = []
mode = 'release'

if ARGUMENTS.get('mode'):
  mode = ARGUMENTS['mode']
if mode == 'release':
  flags = ['-O3']
elif mode == 'debug':
  flags = ['-g']
elif mode == 'emurelease':
  flags = ['-O3']
  nvflags = ['-deviceemu']
elif mode == 'emudebug':
  flags = ['-g']
  nvflags = ['-deviceemu']

env.Append(CFLAGS = flags)
env.Append(CXXFLAGS = flags)
env.Append(NVCCFLAGS = nvflags)
env.Append(HIPFLAGS = flags)

# Now create the program program
sources = ['main.cu']

env.Program('cuda-template', sources)

