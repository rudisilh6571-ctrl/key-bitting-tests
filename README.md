# KeyForge3D

![KeyForge3D Logo](keyforge3d.png)

**KeyForge3D** is an innovative application that extracts the shape of a physical key from a photo, generates a 3D model of the key, and exports it as an STL file for 3D printing. This tool is designed for locksmiths, hobbyists, or anyone needing a quick way to replicate a key using a 3D printer. The app features a user-friendly GUI built with Tkinter, making it easy to upload an image, process the key, and generate a 3D model.

## Features

- **Key Shape Extraction**: Upload a photo of a key to extract its 2D profile.
- **Bitting Analysis**: Automatically detect and analyze the key's bitting pattern (e.g., "05624").
- **3D Model Generation**: Convert the 2D key profile into a 3D model with accurate cuts and dimensions.
- **STL Export**: Export the 3D model as an STL file, ready for 3D printing.
- **User-Friendly GUI**: Simple interface to upload images and generate models with a single click.
- **Scalability**: Scale the key to real-world dimensions from its measured length (falls back to 1 pixel = 0.1 mm).

## Demo

![KeyForge3D Sample](sample.png)

The app processes a photo of a key, extracts its shape, and generates a 3D model like the one shown above. The bitting code (e.g., "05624") is also displayed for reference.

## Installation

### Prerequisites

- **Python 3.8+**: Ensure Python is installed on your system.
- **pip**: Python package manager for installing dependencies.
- **3D Printer (Optional)**: To print the generated key models.

### Dependencies

KeyForge3D relies on the following Python libraries:

- `opencv-python-headless`: For image processing and key shape extraction.
- `numpy`: For numerical operations.
- `trimesh`, `mapbox-earcut`, `manifold3d`: For 3D model generation, groove cutting and STL export.
- `shapely`: For polygon operations.
- `pillow`: For image display in the GUI.

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

The GUI also needs Tk. The python.org installers for Windows and macOS include it. On Linux, install your distribution's Tk package (e.g. `sudo apt install python3-tk` or `sudo pacman -S tk`). Alternatively, on Linux or macOS, `./run.sh` creates a virtualenv with [uv](https://docs.astral.sh/uv/)'s managed Python, which includes Tk.

### Clone the Repository

Clone the KeyForge3D repository to your local machine:

```bash
git clone https://github.com/makalin/KeyForge3D.git
cd KeyForge3D
```

## Usage

1. **Take a good photo**:
   - Put the key flat on plain paper and shoot from **directly above**, filling most of the frame. Leave a margin of paper on every side.
   - Use soft, even light (e.g. near a window, no flash). Hard shadows along the key's edge get counted as part of the key and make cuts look shallower.
   - A flatbed scanner gives the best results.
   - Measure the key's overall length (tip to end of the bow) with a ruler or calipers. The photo alone can't tell how big the key is.

2. **Run the App**:

   ```bash
   ./run.sh            # or: python keyforge3d.py
   ```

   - Click "Upload Key Image" and pick the photo.
   - Enter the **key length** in mm. You can also set the thickness, the number of cuts (0 = detect automatically), and the depth step for your key system.
   - Click "Process Key and Generate 3D Model", then choose where to save the STL. The app shows the detected outline, the toothed region, and each cut it found, so you can check the detection before printing.

3. **Command line** (no GUI):

   ```bash
   ./run.sh photo.jpg --length 50 -o key.stl --overlay check.png
   ./run.sh --help
   ```

4. **3D Printing**:
   - Open the STL in your slicer (e.g. Cura, PrusaSlicer, Bambu Studio).
   - Print flat with a fine layer height (0.1 mm) and a strong material. Plastic keys are good for test fits and emergencies, not daily use.

### How it works and its limits

- The model is the key's photographed silhouette extruded to the given thickness, so the cuts in the STL are the real cuts from the photo.
- A photo can't show groove depths, so enter the grooves along the blade (the keyway profile) from caliper measurements: `face from-to depth`, in mm, measured across the blade from the **spine** (the straight, uncut back edge). `front` is the side facing the camera in the photo and `back` is the other side. Separate grooves with `;` in the GUI, or pass `-g` once per groove on the command line:

  ```bash
  ./run.sh photo.jpg --length 44.9 --thickness 1.9 -g "front 0.8-2.0 0.6" -g "back 3.0-4.6 0.5"
  ```

- Shadows and a slightly tilted phone make the outline read too wide on the shadowed side, and that is usually the toothed edge. To correct it, measure the key's width across (perpendicular to the blade) with calipers at two or more places, and give each as `distance from the bow end:width`. For example, the bow's widest point and the narrowest point next to it: `--check 9:20.45 --check 20:7.44`. KeyForge3D then fixes the scale across the key and trims the shadow off the darker side. The length still sets the scale along the key.

  Grooves are cut square-bottomed, from the bow to the tip. The check image outlines them (magenta = front, orange = back), and the app prints the blade width so you can sanity-check your numbers.
- The bitting code is an **estimate**: cut depth ÷ depth step. It depends on the depth step (Schlage 0.381 mm, Kwikset 0.584 mm) and on the photo's accuracy.

## Code Overview

The main script (`keyforge3d.py`) includes:

- **GUI**: Built with Tkinter for user interaction.
- **Image Processing**: Uses OpenCV to detect the key's outline and analyze its bitting pattern.
- **3D Model Generation**: Uses Trimesh to create a 3D model and export it as an STL file.
- **Error Handling**: Includes basic error handling for invalid images or processing failures.

## Roadmap

- [ ] Add a 3D model preview within the app.
- [x] Scaling from a measured key length.
- [ ] Support for scaling using a reference object (e.g., a coin).
- [ ] Add key type identification (e.g., Schlage SC4, Kwikset KW1).
- [ ] Improve bitting analysis accuracy with machine learning.
- [ ] Create a mobile app version with camera integration.
- [ ] Add support for more key profiles and bitting standards.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the project's coding style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by locksmith tools for key bitting analysis.
- Built with the help of open-source libraries like OpenCV, Trimesh, and Tkinter.

## Contact

For questions, suggestions, or feedback, feel free to open an issue or reach out.
