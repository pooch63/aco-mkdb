#=
Stub provider template. Copy to providers/<name>.jl, implement download + convert,
then `include` it from providers/registry.jl.

Example registration:

  function my_download_raw!(dataset_name, dest_dir)
      # Downloads.download(url, path); maybe gunzip_file!(...)
      return path_to_raw_file
  end

  function my_convert!(raw_path, interactions_csv)
      # Write user_id,item_id,timestamp CSV to interactions_csv
      return nothing
  end

  register_provider!(ProviderAdapter("myprovider", my_download_raw!, my_convert!))

  # If download is unsupported, pass `nothing` for download_raw!:
  # register_provider!(ProviderAdapter("myprovider", nothing, my_convert!))
=#
