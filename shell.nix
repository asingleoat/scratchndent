{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      opencv4
      tifffile
      numpy
      pillow
      scikit-image
    ]))
    exiftool
    imagemagick
  ];
}
