# Zenodo Upload Instructions
**Quick reference for compressing and uploading the dataset**

---

## Step 1: Final Pre-Upload Checks

Before compressing, fill in these placeholders:

### In LICENSE.txt
Replace `[Author Name]` with your name (lines need updating)

### In CITATION.cff
Replace:
- `[Last Name]` with your last name
- `[First Name]` with your first name
- `[ORCID-ID]` with your ORCID (or remove the orcid line if you don't have one)

Example:
```yaml
authors:
  - family-names: "Smith"
    given-names: "Jane"
    orcid: "https://orcid.org/0000-0002-1234-5678"
```

---

## Step 2: Clean Temporary Files

```powershell
cd C:\Users\User\ProjectVeil\Veil\RD_physics\RD_galaxy_phase_transition_dataset_v1.0

# Remove test output
Remove-Item test_output.txt -ErrorAction SilentlyContinue

# Remove Python cache
Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter *.pyc | Remove-Item -Force
```

---

## Step 3: Create Archive

### Option A: ZIP (Windows-friendly)
```powershell
cd C:\Users\User\ProjectVeil\Veil\RD_physics

# Compress (takes ~30 seconds)
Compress-Archive -Path RD_galaxy_phase_transition_dataset_v1.0 -DestinationPath RD_galaxy_phase_transition_v1.0.zip -Force

# Verify size
(Get-Item RD_galaxy_phase_transition_v1.0.zip).Length / 1MB
# Expected: 40-80 MB
```

### Option B: tar.gz (Linux-friendly, better compression)
```powershell
cd C:\Users\User\ProjectVeil\Veil\RD_physics

# Requires tar (built into Windows 10+)
tar -czf RD_galaxy_phase_transition_v1.0.tar.gz RD_galaxy_phase_transition_dataset_v1.0

# Verify size
(Get-Item RD_galaxy_phase_transition_v1.0.tar.gz).Length / 1MB
# Expected: 35-70 MB
```

**Recommendation:** Use ZIP for maximum compatibility with Windows users.

---

## Step 4: Test Archive Extraction (Optional but Recommended)

```powershell
# Create test extraction directory
mkdir C:\Users\User\ProjectVeil\Veil\RD_physics\test_extraction
cd test_extraction

# Extract
Expand-Archive ..\RD_galaxy_phase_transition_v1.0.zip -DestinationPath .

# Quick test
cd RD_galaxy_phase_transition_dataset_v1.0
python -c "import pandas as pd; df = pd.read_csv('data/results/sparc_spirals/tdgl_fits.csv'); print(f'Loaded {len(df)} galaxies')"
# Expected: "Loaded 109 galaxies"

# Clean up
cd ..\..\..
Remove-Item test_extraction -Recurse -Force
```

---

## Step 5: Upload to Zenodo

### 5.1 Create Account
1. Go to https://zenodo.org/
2. Click "Sign up" (top right)
3. Options:
   - Sign up with email
   - OR use GitHub account (recommended - links datasets to your repos)
   - OR use ORCID

### 5.2 Start New Upload
1. Log in to Zenodo
2. Click "Upload" → "New Upload"
3. You'll see the upload form

### 5.3 Upload File
1. Drag and drop `RD_galaxy_phase_transition_v1.0.zip`
2. OR click "Choose files" and select the ZIP
3. Wait for upload to complete (~2-5 minutes depending on internet speed)
4. Verify file size displayed correctly (~50-80 MB)

### 5.4 Fill Metadata Form

**Basic Information:**
- **Upload type:** Dataset
- **Publication date:** December 6, 2025 (or today's date)
- **Title:** 
  ```
  RD Galaxy Phase Transition Dataset: Discovery of a Phase Transition in Galactic Dynamics (v1.0)
  ```

**Creators:**
- Click "Add creator"
- **Name:** [Your name in format: Last, First]
- **Affiliation:** [Your institution]
- **ORCID:** [Your ORCID if you have one]

**Description:**
```
This dataset contains the complete analysis pipeline and results for validating 
Recursive Dynamics (RD) cosmology through galaxy observations. The central 
discovery is a phase transition in galactic dynamics at τ = 0, where τ is the 
unified dynamical timescale.

The dataset includes:
- 109 SPARC spiral galaxies with TDGL rotation curve fits
- 89 ATLAS³D elliptical galaxies with TDGL-Jeans dispersion fits
- Complete analysis scripts for reproducibility (Python)
- 5 publication-quality figures
- Comprehensive documentation

Key Result: 100% of spirals show τ > 0 (active, expanding coherence) while 
87.6% of ellipticals show τ < 0 (frozen, contracting coherence), with S0 
galaxies at the phase boundary. This phase transition is predicted by RD 
cosmology but has no analog in ΛCDM or MOND.

Citation: If you use this dataset, please cite the accompanying paper:
[Your Name] (2026). "Discovery of a Phase Transition in Galactic Dynamics." 
Nature Astronomy (submitted).
```

**Keywords:** (Add these one at a time)
- galaxy dynamics
- cosmology
- rotation curves
- velocity dispersion
- phase transition
- SPARC
- ATLAS3D
- recursive dynamics
- dark matter alternative
- modified gravity

**Additional Descriptions:**
- **Version:** 1.0
- **Language:** English
- **Resource type:** Dataset

**License:**
- Select "Other (Open)" from dropdown
- In the text box:
  ```
  MIT License (for code) + Creative Commons Attribution 4.0 International (CC BY 4.0) (for data)
  See LICENSE.txt in the archive for full terms.
  ```

**Funding:** (if applicable)
- Add funding sources if you have grants to acknowledge

**Related/alternate identifiers:**
Add these related works:
1. **Related identifier:** `10.3847/0004-6256/152/6/157`
   - **Relation:** Is supplement to / Cites
   - **Scheme:** DOI
   - **Description:** SPARC catalog (Lelli et al. 2016)

2. **Related identifier:** `10.1111/j.1365-2966.2010.18174.x`
   - **Relation:** Is supplement to / Cites
   - **Scheme:** DOI
   - **Description:** ATLAS³D survey (Cappellari et al. 2011)

3. **Related identifier:** [Your GitHub repo URL if you have one]
   - **Relation:** Is derived from
   - **Scheme:** URL
   - **Description:** Source code repository

**Contributors:** (Optional)
- SPARC Team (Data Collector)
- ATLAS³D Collaboration (Data Collector)

**Subjects:**
- Astrophysics
- Cosmology
- Galaxy Dynamics

**Communities:** (Optional - helps discoverability)
- Search and add:
  - "Astronomy"
  - "Astrophysics"
  - "Open Science"

### 5.5 Save and Review
1. Click "Save" (bottom of form)
2. This creates a DRAFT (not published yet)
3. Review all information carefully
4. Check the preview of your dataset page

### 5.6 Test Download (Recommended)
1. From your draft, click "Preview"
2. Click the download link to test
3. Extract and verify contents
4. If anything is wrong, you can edit the draft

### 5.7 Publish
1. Once everything looks good, click "Publish"
2. **WARNING:** After publishing, the dataset is PERMANENT and cannot be deleted
   - You can upload new versions, but old versions remain
   - You CAN edit metadata after publishing
3. Confirm the publication
4. **You'll receive a DOI immediately** (something like `10.5281/zenodo.XXXXXXX`)

---

## Step 6: Post-Publication Tasks

### 6.1 Get Your DOI
After publishing, Zenodo displays your DOI. Copy it (format: `10.5281/zenodo.XXXXXXX`)

### 6.2 Update CITATION.cff with Real DOI
```powershell
cd C:\Users\User\ProjectVeil\Veil\RD_physics\RD_galaxy_phase_transition_dataset_v1.0

# Edit CITATION.cff
# Replace [zenodo-doi] with your actual DOI (e.g., 10.5281/zenodo.1234567)
```

**Note:** You can upload a new version to Zenodo with the updated CITATION.cff, or just note this for future versions.

### 6.3 Update Your Manuscript
Add to "Data Availability" section:
```
Data and analysis code are available on Zenodo: 
https://doi.org/10.5281/zenodo.XXXXXXX (replace with your DOI)
```

Add to References:
```
[Your Name]. (2025). RD Galaxy Phase Transition Dataset: Discovery of a 
Phase Transition in Galactic Dynamics (v1.0) [Data set]. Zenodo. 
https://doi.org/10.5281/zenodo.XXXXXXX
```

### 6.4 Share Your Dataset
- **Twitter/X:** "New dataset out! 194 galaxies reveal phase transition in galactic dynamics. Data & code on @Zenodo: [DOI link]"
- **LinkedIn:** Share with research community
- **ResearchGate:** Add to your publications
- **Reddit:** r/Astrophysics, r/DataIsBeautiful (with Figure 5)

---

## Troubleshooting

### Upload Fails or Times Out
- **Solution:** Check internet connection, try again
- **Alternative:** Split into smaller archives if size is an issue (shouldn't be needed for 50-80 MB)

### Metadata Form is Confusing
- **Help:** Click the "?" icons next to each field for Zenodo's help text
- **Example:** Look at similar datasets on Zenodo for formatting examples

### Forgot to Fill Author Name
- **Solution:** After publishing, you can edit metadata
- **Steps:** Go to your upload → Click "Edit" → Update creators → Save

### Want to Upload New Version Later
- **Solution:** Zenodo supports versioning
- **Steps:** 
  1. Go to your published dataset
  2. Click "New version"
  3. Upload new files
  4. Update version number (e.g., v1.0 → v1.1)
  5. Describe changes in metadata
  6. Publish new version (gets new DOI, old version remains accessible)

### Need to Cite Different Papers
- **Solution:** Add more related identifiers in metadata
- **Example:** If your paper gets accepted, add its DOI as a related identifier

---

## Timeline

**Estimated time breakdown:**
- Metadata filling: 10-15 minutes
- File upload: 3-5 minutes (depends on internet speed)
- Review and test: 5-10 minutes
- **Total:** 20-30 minutes from start to published DOI

---

## After Upload

### What Zenodo Provides
- ✅ Permanent DOI for citation
- ✅ Long-term preservation (CERN-hosted, backed up)
- ✅ Version control (can upload v1.1, v1.2, etc.)
- ✅ Usage statistics (downloads, views)
- ✅ Integration with GitHub (can auto-archive GitHub releases)
- ✅ Automatic indexing in Google Dataset Search
- ✅ FAIR principles compliance (Findable, Accessible, Interoperable, Reusable)

### Citation Tracking
- Your dataset will appear in:
  - Google Scholar (when papers cite your DOI)
  - ADS (Astrophysics Data System) - register separately
  - Web of Science / Scopus (if papers cite it)

### Future Updates
If you need to update the dataset:
1. Go to Zenodo upload
2. Click "New version"
3. Upload updated files
4. Document changes in version notes
5. New DOI assigned (old version still accessible)

---

## Questions?

**Zenodo Support:**
- Help docs: https://help.zenodo.org/
- Email: info@zenodo.org

**Dataset Issues:**
- GitHub Issues: [Your repo]/issues
- Email: [Your contact email]

---

**Ready to upload?** Follow steps 1-6 above. Good luck! 🚀

---

**Generated:** December 6, 2025  
**Dataset Version:** 1.0  
**Guide Version:** 1.0
