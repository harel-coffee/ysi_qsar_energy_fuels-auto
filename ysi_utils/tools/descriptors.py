import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from collections import Counter
from jinja2 import Template

from .sh_utils import make_temp_directory, run_process

descList = dict(Descriptors.descList)
form_parse = re.compile('([A-Z][a-z]?)(\d*)')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_descriptors(smiles):
    """ Use RDkit to get molecular descriptors for the given smiles string """

    mol = Chem.MolFromSmiles(smiles)
    return pd.Series({name: func(mol) for name, func in descList.items()})


def get_element_dict(smiles):
    """ Given a smiles string, return a dictionary of the of elements by count

    """

    mol = Chem.MolFromSmiles(smiles)
    mol2 = Chem.AddHs(mol)
    c = Counter(atom.GetSymbol() for atom in mol2.GetAtoms())
    return pd.Series(c)


def get_DBE(smiles):
    """Use RDkit to find atoms in molecule and then calculate DBE as seen in
    doi:10.1016/j.combustflame.2016.01.034

    by: Paul Kairys
    date: 6/7/16

    """
    if smiles is None:
        return None

    elem_num = dict(C=0, N=0, H=0)
    elem_num.update(get_element_dict(smiles))

    # DBE calculated by eq. 2 in
    # Barrientos, E. J., Anderson, J. E., Maricq, M. M., & Boehman, A. L.
    # (2016) Particulate matter indices using fuel smoke point for vehicle
    # emissions with gasoline, ethanol blends, and butanol blends, 1-12.
    # http://doi.org/10.1016/j.combustflame.2016.01.034
    DBE = (2 * elem_num['C'] + 2 - elem_num['H'] + elem_num['N']) / 2
    return DBE


def structure_optimization(smiles, mol_cas=None):
    """Given an input smiles string, create and optimize a 3D geometry
    representation. mol_id is an optional string to attach as a property to the
    mol.

    Returns an RDKit mol object

    """
    mol = Chem.MolFromSmiles(smiles)
    molH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(molH)
    AllChem.MMFFOptimizeMolecule(molH)
    Chem.rdPartialCharges.ComputeGasteigerCharges(molH)

    molH.SetProp('SMILES', smiles)

    if mol_cas:
        molH.SetProp('CAS', mol_cas)

    return molH


def write_sdf(smiles_list, cas_list=None, output_file='output.sdf'):
    """Given an iterable of SMILES strings, save optimized 3D geometries to the
    given SDF file

    """
    mol_writer = Chem.SDWriter(output_file)

    try:
        iter(cas_list)
    except TypeError:
        cas_list = [None] * len(smiles_list)

    if tqdm:
        for smiles, cas in tqdm(
                zip(smiles_list, cas_list), total=len(smiles_list)):

            mol = structure_optimization(smiles, mol_cas=cas)
            mol_writer.write(mol)

    else:

        for smiles, cas in zip(smiles_list, cas_list):
            mol = structure_optimization(smiles, mol_cas=cas)
            mol_writer.write(mol)

    mol_writer.close()


def calculate_dragon_descriptors(smiles_list, verbose=False):
    """Calculate dragon descriptors for the given molecules. Assumes
    `dragon7shell` is in the current path. (I.e., this is being run on skynet)
    """

    drt = Template(dragon_template)

    with make_temp_directory('dragon') as temp_dir:

        input_path = temp_dir + '/dragon_input.sdf'
        output_path = temp_dir + '/dragon_output.tsv'
        script_path = temp_dir + '/dragon_script.drt'

        write_sdf(smiles_list,  output_file=input_path)

        with open(script_path, 'w') as f:
            f.write(drt.render(input=input_path, output=output_path))

        run_process('dragon7shell -s ' + script_path, verbose=verbose)

        dragon_descriptors = pd.read_csv(output_path, sep='\t', index_col=0)
        dragon_descriptors.index = smiles_list
        dragon_descriptors.drop('NAME', 1, inplace=True)

        return dragon_descriptors


dragon_template = \
    """<?xml version="1.0" encoding="utf-8"?>
<DRAGON version="7.0.0" description="AllDescriptorsCalculation"
        script_version="1" generation_date="13/06/2016">
  <OPTIONS>
    <Decimal_Separator value="."/>
    <Missing_String value="NaN"/>
    <RejectDisconnectedStrucuture value="true"/>
    <RejectUnusualValence value="true"/>
    <SaveOnlyData value="false"/>
  </OPTIONS>
  <MOLFILES>
        <molInput value="file"/>
        <molFile value="{{ input }}"/>
  </MOLFILES>
    <OUTPUT>
        <SaveStdOut value="false"/>
        <SaveFile value="true"/>
        <SaveType value="singlefile"/>
        <SaveFilePath value="{{ output }}"/>
        <logMode value="stderr"/>
    </OUTPUT>
  <DESCRIPTORS>
  {% for n in range(1,31) %}
    <block id="{{n}}"  SelectAll="true"/>
  {% endfor %}
  </DESCRIPTORS>
</DRAGON>"""
