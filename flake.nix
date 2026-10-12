{
  description = "Bulk image generation with FLUX on RTX 5070 Ti";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          # why not all archs: nccl/nvshmem would compile for every GPU
          cudaCapabilities = [ "12.0" ];
        };
      };
      # why not source torch: CUDA build isn't on the binary cache, compiles for hours
      python = pkgs.python3.override {
        packageOverrides = _: prev: {
          torch = prev.torch-bin.override { cudaPackages = pkgs.cudaPackages_13; };
          triton = prev.triton-bin.override { cudaPackages = pkgs.cudaPackages_13; };
        };
      };
      env = python.withPackages (
        ps: with ps; [
          diffusers
          transformers
          accelerate
          sentencepiece
          safetensors
          pillow
          huggingface-hub
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ env pkgs.uv ];
      };
    };
}
