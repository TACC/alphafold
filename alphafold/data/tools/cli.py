# -------------------- Functional interface for Click CLI ----------------------
import os
import json
import click
from collections import OrderedDict
import pickle
from functools import lru_cache
import hashlib
from typing import Mapping, Optional, Sequence
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer as jackhmmer_wrapper
from alphafold.data import parsers, templates, pipeline
from alphafold.data.tools.cache_utils import cache_to_pckl
import logging

logging.basicConfig(level=logging.DEBUG)

@click.group()
def cli():
    pass


def parse_fasta_path(input_fasta_path):
  """Given fasta file at `input_fasta_path`, calls `parsers.parse_fasta`
  returning tuple of input sequence, input description, and sequence length.
  """
  with open(input_fasta_path) as f:
      input_fasta_str = f.read()
  input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
  if len(input_seqs) != 1:
    raise ValueError(
        f'More than one input sequence found in {input_fasta_path}.')
  return (input_seqs[0], input_descs[0], len(input_seqs[0]))


def write_output(data, fname, output_dir) -> str:
  fp = os.path.join(output_dir, fname)
  with open(fp, 'w') as f:
    f.write(data)
  return fp


# ------------------------------------------------------------------------------

@cli.command(name='jackhmmer')
@click.option('--input-fasta-path', required=True, type=click.Path())
@click.option('--jackhmmer-binary-path', required=True, type=click.Path())
@click.option('--database-path', required=True, type=click.Path())
@click.option('--output-dir', required=True, type=click.Path(file_okay=False))
def jackhmmer_cli(*args, **kwargs):
    return jackhmmer(*args, **kwargs)


@cache_to_pckl(exclude_kw=['output_dir', 'fname'])
def jackhmmer(input_fasta_path: str, jackhmmer_binary_path: str,
              database_path: str, fname: str, output_dir: str):
  jackhmmer_uniref90_runner = jackhmmer_wrapper.Jackhmmer(
      binary_path=jackhmmer_binary_path,
      database_path=database_path)
  result = jackhmmer_uniref90_runner.query(input_fasta_path)[0]['sto']
  return write_output(result, fname, output_dir=output_dir)


@cache_to_pckl(exclude_kw='output_dir')
def hhsearch_pdb70(jackhmmer_uniref90_hits_path, hhsearch_binary_path: str,
                   pdb70_database_path: str, output_dir: str,
                   uniref_max_hits):
  with open(jackhmmer_uniref90_hits_path, 'r') as f:
    jackhmmer_uniref90_hits = f.read()
  hhsearch_pdb70_runner = hhsearch.HHSearch(
      binary_path=hhsearch_binary_path,
      databases=[pdb70_database_path])
  uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
      jackhmmer_uniref90_hits, max_sequences=uniref_max_hits)
  result = hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
  return write_output(result, 'pdb70_hits.hhr', output_dir=output_dir)


@cache_to_pckl(exclude_kw='output_dir')
def hhblits(input_fasta_path: str, hhblits_binary_path: str,
            bfd_database_path: str, uniclust30_database_path: str,
            output_dir: str):
  hhblits_bfd_uniclust_runner = hhblits.HHBlits(
      binary_path=hhblits_binary_path,
      databases=[bfd_database_path, uniclust30_database_path])
  result = hhblits_bfd_uniclust_runner.query(input_fasta_path)['a3m']
  return write_output(result, 'bfd_uniclust_hits.a3m', output_dir=output_dir)


@cache_to_pckl()
def template_featurize(input_fasta_path, hhsearch_hits_path, mmcif_dir: str,
                       max_template_date, max_hits, kalign_binary_path,
                       release_dates_path, obsolete_pdbs_path,
                       strict_error_check):
  with open(hhsearch_hits_path, 'r') as f:
    hhsearch_hits = parsers.parse_hhr(f.read())
  template_featurizer = templates.TemplateHitFeaturizer(
    mmcif_dir=mmcif_dir,
    max_template_date=max_template_date,
    max_hits=max_hits,
    kalign_binary_path=kalign_binary_path,
    release_dates_path=release_dates_path,
    obsolete_pdbs_path=obsolete_pdbs_path,
    strict_error_check=strict_error_check,
  )
  input_sequence, _, _ = parse_fasta_path(input_fasta_path)
  features = template_featurizer.get_templates(
      query_sequence=input_sequence,
      query_pdb_code=None,
      query_release_date=None,
      hits=hhsearch_hits).features
  logging.info('Total number of templates (NB: this can include bad '
                'templates and is later filtered to top 4): %d.',
                features['template_domain_names'].shape[0])
  return features


@cache_to_pckl()
def make_msa_features(jackhmmer_uniref90_hits_path, jackhmmer_mgnify_hits_path, 
                      bfd_hits_path, mgnify_max_hits: int, use_small_bfd: bool):
    with open(jackhmmer_uniref90_hits_path, 'r') as f:
      uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(f.read())

    with open(jackhmmer_mgnify_hits_path, 'r') as f:
      mgnify_msa, mgnify_deletion_matrix, _  = parsers.parse_stockholm(f.read())

    with open(bfd_hits_path, 'r') as f:
      bfd_hits = f.read()

    mgnify_msa = mgnify_msa[:mgnify_max_hits]
    mgnify_deletion_matrix = mgnify_deletion_matrix[:mgnify_max_hits]

    if use_small_bfd:
      bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(bfd_hits)
    else:
      bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(bfd_hits)

    msa_features = pipeline.make_msa_features(
        msas=(uniref90_msa, bfd_msa, mgnify_msa),
        deletion_matrices=(uniref90_deletion_matrix,
                           bfd_deletion_matrix,
                           mgnify_deletion_matrix))

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    return msa_features


if __name__ == '__main__':
    cli()