#!/usr/bin/env python3
"""Fixed-geometry torsion-space basin-width analysis for an A' PoC site."""
from __future__ import annotations
import argparse, math
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import ttest_rel
from qfit.solvers import get_qp_solver_class
from run_d1_8d_sequential_poc import atomic_csv, atomic_json, rmsd
from run_d1_aprime_sequential import APrimeSequential, seam_vector, internal_geometry
from run_d1_aprime_representability_gate import Gate

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--pdb-id',default='7UTC'); p.add_argument('--chain',default='A'); p.add_argument('--resnum',type=int,default=52)
 p.add_argument('--sequential-output',type=Path,default=Path('/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_sequential_v3'))
 p.add_argument('--moving-slot',type=int,choices=(1,2),default=2,
                help='Scan this converged slot toward its corresponding deposited state.')
 p.add_argument('--full-objective',action='store_true',help='Also score density + frozen-AL seam + Rama + omega planarity.')
 a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 runner=APrimeSequential(a.output,80,6,a.pdb_id,a.chain,a.resnum); base=runner.base
 final=np.load(a.sequential_output/'final_slots.npz')
 slot1,slot2=final['slot1_window'],final['slot2_window']
 gate=Gate(a.pdb_id,a.chain,a.resnum)
 endpoint_parameters = (np.zeros(gate.rotator.ndofs) if a.moving_slot == 1
                        else np.asarray(gate.solve(np.zeros(gate.rotator.ndofs))['parameters_deg'],float))
 endpoint_name = 'A' if a.moving_slot == 1 else 'B'
 moving_slot = slot1 if a.moving_slot == 1 else slot2
 # Fit the chosen recovered slot to the same 20-parameter A-prime chart.
 target=moving_slot[runner.bb_indices]
 recovered=least_squares(lambda q:(runner.forward(q)[runner.bb_indices]-target).ravel(),np.zeros(20),method='lm',max_nfev=5000,ftol=1e-12,xtol=1e-12,gtol=1e-12).x
 a_b=np.asarray(base.a_residue.b,float); by_name=dict(zip(base.b_residue.name.tolist(),base.b_residue.b)); b_b=np.asarray([by_name[x] for x in base.central.name.tolist()],float)
 def density(coords,b):
  central=base.central_coordinates(coords); d=next(base.qfit._transformer.get_conformers_densities([central],[b]))[base.mask].astype(float,copy=False)
  return np.maximum(d,base.qfit.options.bulk_solvent_level)
 y=base.target; coords=np.argwhere(base.mask)*np.asarray(base.qfit.xmap.voxelspacing,float); n=len(y); nt=round(.2*n); rng=np.random.default_rng(20260805)
 for _ in range(5): rng.choice(n,size=nt,replace=False) # reproduce random-split draws before blocked directions
 splits=[]
 for _ in range(5):
  d=rng.normal(size=3); d/=np.linalg.norm(d); test=np.sort(np.argsort(coords@d)[:nt]); splits.append((np.setdiff1d(np.arange(n),test,assume_unique=True),test))
 fractions=np.round(np.arange(-.4,1.6001,.1),1); rows=[]; states={}
 final_result=json.loads((a.sequential_output/'result.json').read_text())
 moving_config=final_result['slots'][f'slot{a.moving_slot}']['convergence']
 full_normalizer=float(moving_config['normalizer_initial_rss'])
 frozen_lambdas=np.asarray(moving_config['final_lambdas'],float)
 for f in fractions:
  q=(1-f)*recovered+f*endpoint_parameters; moving=runner.forward(q)
  if a.moving_slot == 1:
   matrix=np.vstack((density(moving,a_b),density(slot2,b_b)))
  else:
   matrix=np.vstack((density(slot1,a_b),density(moving,b_b)))
  g,t,r=seam_vector(runner.initial_backbone,moving[runner.bb_indices]); omega,omega_delta,rama,_=runner.omega_and_rama(moving); geom=internal_geometry(runner.window,runner.initial,moving)
  seam_energy=float(runner.rho/2*np.square(g+frozen_lambdas/runner.rho).sum())
  rama_barrier=np.maximum(0.0,np.log(runner.rama_floor/np.maximum(np.asarray(rama),1e-12)))
  rama_energy=float(runner.rama_weight*np.square(rama_barrier).sum())
  planar_energy=float(runner.planar_weight*np.square(omega_delta/runner.omega_scale_deg).sum())
  state={'fraction':float(f),'slot2_rmsd_to_A_A':float(rmsd(base.central_backbone(moving),base.a_backbone)),'slot2_rmsd_to_B_A':float(rmsd(base.central_backbone(moving),base.b_backbone)),'seam_A_equivalent':g.tolist(),'seam_norm_A_equivalent':float(np.linalg.norm(g)),'seam_translation_norm_A':float(np.linalg.norm(t)),'seam_rotation_norm_deg':float(np.linalg.norm(np.degrees(r))),'min_rama_probability':float(min(rama)),'max_bond_length_change_A':geom['max_abs_bond_length_change_from_A_A'],'max_bond_angle_change_deg':geom['max_abs_bond_angle_change_from_A_deg'],'AL_seam_energy_frozen_multipliers':seam_energy,'rama_energy':rama_energy,'omega_planarity_energy':planar_energy,'heldout':[],'full_objective_heldout':[],'coordinates':moving}
  for repeat,(train,test) in enumerate(splits):
   solver=get_qp_solver_class('CVXPYSolver')(y[train],matrix[:,train]); solver.solve_qp(); w=np.asarray(solver.weights,float); rss=float(np.square(y[test]-w@matrix[:,test]).sum()); state['heldout'].append(rss)
   full=float((n/len(test))*rss/full_normalizer + seam_energy + rama_energy + planar_energy)
   state['full_objective_heldout'].append(full)
   rows.append({'fraction':float(f),'repeat':repeat,'heldout_rss':rss,'full_objective_heldout':full,'occupancies_train':w.tolist()})
  states[float(f)]=state
 best=min(states,key=lambda f:np.mean(states[f]['heldout']))
 best_values=np.asarray(states[best]['heldout']); full_best=min(states,key=lambda f:np.mean(states[f]['full_objective_heldout']))
 full_best_values=np.asarray(states[full_best]['full_objective_heldout']); summary=[]
 for f in fractions:
  s=states[float(f)]; diff=np.asarray(s['heldout'])-best_values; mean=float(diff.mean()); sd=float(diff.std(ddof=1)); pval=1.0 if f==best else float(ttest_rel(s['heldout'],best_values).pvalue)
  full_diff=np.asarray(s['full_objective_heldout'])-full_best_values; full_mean=float(full_diff.mean()); full_sd=float(full_diff.std(ddof=1)); full_pval=1.0 if f==full_best else float(ttest_rel(s['full_objective_heldout'],full_best_values).pvalue)
  summary.append({k:v for k,v in s.items() if k not in ('heldout','full_objective_heldout','coordinates')}|{'heldout_rss_mean':float(np.mean(s['heldout'])),'heldout_rss_sd':float(np.std(s['heldout'],ddof=1)),'paired_difference_vs_best_per_split':diff.tolist(),'paired_difference_mean':mean,'paired_difference_sd':sd,'paired_ttest_pvalue':pval,'within_one_paired_sd':bool(abs(mean)<=sd) if f!=best else True,'paired_not_significant_p_ge_0p05':bool(pval>=.05),'full_objective_heldout_mean':float(np.mean(s['full_objective_heldout'])),'full_objective_heldout_sd':float(np.std(s['full_objective_heldout'],ddof=1)),'full_objective_paired_difference_vs_best_per_split':full_diff.tolist(),'full_objective_paired_difference_mean':full_mean,'full_objective_paired_difference_sd':full_sd,'full_objective_paired_ttest_pvalue':full_pval,'full_objective_within_one_paired_sd':bool(abs(full_mean)<=full_sd) if f!=full_best else True,'full_objective_paired_not_significant_p_ge_0p05':bool(full_pval>=.05)})
 def band(key): return [x['fraction'] for x in summary if x[key]]
 result={'status':'complete','site':f'{a.pdb_id}_{a.chain}_{base.a_residue.resn[0]}{a.resnum}','operation':'torsion-space interpolation and fixed-geometry QP occupancy fitting only; no density-driven coordinate optimisation','moving_slot':a.moving_slot,'fixed_slot':2 if a.moving_slot == 1 else 1,'interpolation_endpoint_deposited_state':endpoint_name,'fair_bfactor_rendering':'slot1 uses deposited-A B factors; slot2 uses deposited-B B factors','endpoint_fit_full_window_rmsd_A':gate.solve(endpoint_parameters)['full_window_backbone_rmsd_A'] if a.moving_slot == 2 else 0.0,'recovered_moving_slot_parameter_fit_rmsd_A':float(rmsd(runner.forward(recovered)[runner.bb_indices],target)),'blocked_splits':5,'best_fraction':float(best),'best_heldout_rss_mean':float(np.mean(best_values)),'within_one_paired_sd_fractions':band('within_one_paired_sd'),'paired_not_significant_fractions':band('paired_not_significant_p_ge_0p05'),'full_objective':{'enabled':bool(a.full_objective),'definition':'5x held-out RSS / moving-slot initial full-mask RSS + rho/2||g+lambda/rho||^2 + Rama barrier + omega planarity; lambda frozen at the converged moving-slot AL value','moving_slot_normalizer_initial_rss':full_normalizer,'frozen_moving_slot_AL_multipliers':frozen_lambdas.tolist(),'best_fraction':float(full_best),'best_mean':float(np.mean(full_best_values)),'within_one_paired_sd_fractions':band('full_objective_within_one_paired_sd'),'paired_not_significant_fractions':band('full_objective_paired_not_significant_p_ge_0p05')},'per_fraction':summary}
 atomic_csv(a.output/'per_fraction_per_split.csv',rows); atomic_json(a.output/'result.json',result); atomic_json(a.output/'progress.json',{'status':'complete','points':len(summary)}); print(result)
if __name__=='__main__': main()
