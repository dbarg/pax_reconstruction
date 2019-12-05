import os

def main():
    
    tmpdir   = os.system.environ.get('TMPDIR') 
    slurmdir = os.system.environ.get('SLURM_TMPDIR') 
    cmd      = 'ls -l {0}'.format(slurmdir)
    
    print('------------')
    print('$SLURM_TMPDIR: {0}'.format(tmpdir))
    print(slurmdir)
    os.system('env')
    os.system(cmd)
    
    return

if __name__ == '__main__':
    main()
