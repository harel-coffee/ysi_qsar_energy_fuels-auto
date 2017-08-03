from rdkit import Chem
from rdkit.Chem import AllChem

from tqdm import tqdm
from collections import Counter

supplier = Chem.SDMolSupplier('ChEBI_complete.sdf')
writer = Chem.SDWriter('chebi_subset.sdf')
atoms_of_interest = {'C', 'H', 'O'}

num_mols = 0

for mol in tqdm(supplier):

    try:
        elements = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        num_carbons = elements['C']

    except AttributeError:
        continue

    # Find out if the molecule contains atoms which are outside 'C', 'H', or 'O'.
    if ((not set(elements.keys()).difference(atoms_of_interest)) &
        (num_carbons <= 12) &
        (num_carbons >= 1)):

        mol.SetProp('_Name', mol.GetProp('ChEBI ID'))
        molH = Chem.AddHs(mol)
        AllChem.EmbedMolecule(molH)
        AllChem.MMFFOptimizeMolecule(molH)

        num_mols += 1
        writer.write(molH)

writer.close()
print("Number of written compounds: {}".format(num_mols))
