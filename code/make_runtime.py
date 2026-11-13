'Needs paths.py; validates explicit CLI and allows writes only to WORK/data. Shared temporal/spatial/production runs.'
from pathlib import Path
import argparse,os,sys
import paths

def cli(doc):
 p=argparse.ArgumentParser(description=doc);p.add_argument('--run',action='store_true',help='Explicitly execute a derivation with upstream inputs');p.add_argument('--work-dir',type=Path,default=paths.WORK);a,remaining=p.parse_known_args()
 if not a.run:p.error('Use --run only after checking upstream paths; --help never reads upstream data')
 paths.WORK=a.work_dir.resolve();paths.DERIVED=paths.WORK/'source_data'
 assert not any(paths.WORK.is_relative_to(x) for x in [paths.RAW,paths.V6,paths.FORMAL,paths.SPATIAL]),'Work directory must be outside upstream namespaces'
 paths.WORK.mkdir(parents=True,exist_ok=True);paths.DERIVED.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(paths.WORK/'.mplconfig');os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
 os.environ['CUDA_VISIBLE_DEVICES']='';sys.argv=[sys.argv[0],*remaining]
 def guard(event,args):
  targets=[]
  if event=='open' and isinstance(args[0],(str,bytes)):
   mode=args[1];flags=args[2];writing=(mode is not None and any(c in mode for c in 'wax+')) or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC));targets=[args[0]] if writing else []
  elif event in ['os.mkdir','os.remove','os.rmdir']:targets=[args[0]]
  elif event in ['os.rename','os.replace']:targets=list(args[:2])
  for target in targets:
   q=Path(os.fsdecode(target)).resolve()
   if len(q.parts)>1 and q.parts[1]=='dev':continue
   if not (q.is_relative_to(paths.WORK) or q.is_relative_to(paths.DATA)):raise PermissionError('Derivation refuses upstream write: '+str(q))
 sys.addaudithook(guard)
