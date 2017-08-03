from rdkit import Chem
from rdkit.Chem import AllChem

supplier = Chem.SmilesMolSupplier('chebi_subset.smi')

for mol in supplier:
    print(Chem.MolToSmiles(mol))
    molH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(molH)
    AllChem.MMFFOptimizeMolecule(molH)
