"""Modular version of alphafold.data.pipeline"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools.cli import *
from alphafold.data.pipeline import make_sequence_features
import numpy as np

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]


@dataclass
class ModularDataPipeline:
  """Modular version of alphafold.data.pipeline.DataPipeline"""
  jackhmmer_binary_path: str
  hhblits_binary_path: str
  hhsearch_binary_path: str
  uniref90_database_path: str
  mgnify_database_path: str
  pdb70_database_path: str
  use_small_bfd: bool

  # for construction of TemplateHitFeaturizer, replacing
  # template_featurizer: templates.TemplateHitFeaturizer
  mmcif_dir: str
  max_template_date: str
  max_hits: int
  kalign_binary_path: str
  release_dates_path: str = None
  obsolete_pdbs_path: str = None
  strict_error_check: bool = False

  mgnify_max_hits: int = 501
  uniref_max_hits: int = 10000
  bfd_database_path: str = None
  uniclust30_database_path: str = None
  small_bfd_database_path: str = None

  def jackhmmer_uniref90(self, input_fasta_path: str):
    return jackhmmer(
      input_fasta_path=input_fasta_path, 
      jackhmmer_binary_path=self.jackhmmer_binary_path, 
      database_path=self.uniref90_database_path, 
      fname='uniref90_hits.sto',
      output_dir=self.msa_output_dir
    )

  def jackhmmer_mgnify(self, input_fasta_path: str):
    return jackhmmer(
      input_fasta_path=input_fasta_path, 
      jackhmmer_binary_path=self.jackhmmer_binary_path, 
      database_path=self.mgnify_database_path, 
      fname='mgnify.sto',
      output_dir=self.msa_output_dir
    )
  
  def hhsearch_pdb70(self, jackhmmer_uniref90_hits_path):
    return hhsearch_pdb70(
      jackhmmer_uniref90_hits_path=jackhmmer_uniref90_hits_path, 
      hhsearch_binary_path=self.hhsearch_binary_path,
      pdb70_database_path=self.pdb70_database_path, 
      uniref_max_hits=self.uniref_max_hits,
      output_dir=self.msa_output_dir
    )
  
  def jackhmmer_small_bfd(self, input_fasta_path):
    return jackhmmer(
      input_fasta_path=input_fasta_path, 
      jackhmmer_binary_path=self.jackhmmer_binary_path, 
      database_path=self.small_bfd_database_path, 
      fname='small_bfd_hits.sto',
      output_dir=self.msa_output_dir
    )

  def hhblits(self, input_fasta_path):
    return hhblits(
      input_fasta_path=input_fasta_path, 
      hhblits_binary_path=self.hhblits_binary_path,
      bfd_database_path=self.bfd_database_path,
      uniclust30_database_path=self.uniclust30_database_path,
      output_dir=self.msa_output_dir
    )

  def template_featurize(self, input_fasta_path, hhsearch_hits_path):
    return template_featurize(
      input_fasta_path=input_fasta_path, 
      hhsearch_hits_path=hhsearch_hits_path,
      mmcif_dir=self.mmcif_dir,
      max_template_date=self.max_template_date, 
      max_hits=self.max_hits, 
      kalign_binary_path=self.kalign_binary_path,
      release_dates_path=self.release_dates_path, 
      obsolete_pdbs_path=self.obsolete_pdbs_path,
      strict_error_check=self.strict_error_check
    )
  
  def make_msa_features(self, jackhmmer_uniref90_hits_path, jackhmmer_mgnify_hits_path,
                        bfd_hits_path):
    return make_msa_features(jackhmmer_uniref90_hits_path, jackhmmer_mgnify_hits_path,
                             bfd_hits_path,
                             mgnify_max_hits=self.mgnify_max_hits,
                             use_small_bfd=self.use_small_bfd)
  
  def make_sequence_features(self, input_fasta_path):
    input_sequence, input_description, num_res = parse_fasta_path(input_fasta_path)
    return make_sequence_features(sequence=input_sequence, 
                                  description=input_description, 
                                  num_res=num_res)

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    self.msa_output_dir = msa_output_dir

    jackhmmer_uniref90_hits_path = self.jackhmmer_uniref90(input_fasta_path)
    hhsearch_hits_path = self.hhsearch_pdb70(jackhmmer_uniref90_hits_path)
    template_features = self.template_featurize(input_fasta_path, hhsearch_hits_path)

    if self.use_small_bfd:
      bfd_hits_path = self.jackhmmer_small_bfd(input_fasta_path)
    else:
      bfd_hits_path = self.hhblits(input_fasta_path)

    jackhmmer_mgnify_hits_path = self.jackhmmer_mgnify(input_fasta_path)
    sequence_features = self.make_sequence_features(input_fasta_path)
    msa_features = self.make_msa_features(jackhmmer_uniref90_hits_path,
                                          jackhmmer_mgnify_hits_path,
                                          bfd_hits_path)
    return {**sequence_features, **msa_features, **template_features}
