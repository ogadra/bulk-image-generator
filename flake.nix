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
      # why not nixpkgs hk and ruff: both lag behind the latest releases
      releases = {
        hk = {
          version = "2.5.0";
          url = "https://github.com/jdx/hk/releases/download/v2.5.0/hk-x86_64-unknown-linux-musl.tar.gz";
          sha256 = "55b349ed8ebbb9cc11e3ff7f741230035061e45621c0b2c87fffeeb37fadcfd0";
        };
        ruff = {
          version = "0.17.0";
          url = "https://github.com/astral-sh/ruff/releases/download/0.17.0/ruff-x86_64-unknown-linux-musl.tar.gz";
          sha256 = "0df9421fd0df9aa7a62dab4d1086e085b7c7888b6343fc5a9c5f630771d61aae";
        };
      };
      lintTools = pkgs.lib.mapAttrsToList (
        name: release:
        pkgs.stdenvNoCC.mkDerivation {
          pname = name;
          inherit (release) version;
          src = pkgs.fetchurl { inherit (release) url sha256; };
          sourceRoot = ".";
          installPhase = ''
            install -Dm755 "$(find . -type f -name ${name})" "$out/bin/${name}"
          '';
        }
      ) releases;
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [
            env
            pkgs.uv
          ]
          ++ lintTools;
        };
        lint = pkgs.mkShell { packages = lintTools; };
      };
    };
}
