#TODO: `compile_readme()`: impelement
#TODO: `current2legacy()`: impelement

#%%imports
using Pkg
using TOML
using UUIDs

#module
include("./src/astroLuStjl.jl")
using .astroLuStjl

#%%definitions
"""
    - function to build the project into an installable module

    Parameters
    ----------
        - `project_toml`
            - `String`, optional
            - path to `Project.toml` file of the module
            - the default is `./Project.toml`

    Raises
    ------

    Returns
    -------

    Dependencies
    ------------
        - `Pkg`
        - `TOML`
        - `UUIDs`

    Comments
    --------

"""
function build_project_toml(
    project_toml::String="./Project.toml",
    )

    #build first template (only contains dependencies)
    Pkg.activate(".")   #activate project env
    Pkg.resolve()       #ensure dependencies are up to date
    Pkg.status()        #list dependencies

    #read file
    toml = TOML.parsefile(project_toml)
    
    #add additional metadata
    toml["name"]            = astroLuStjl.__modulename__
    toml["version"]         = astroLuStjl.__version__
    toml["author"]          = astroLuStjl.__author__
    toml["author_email"]    = astroLuStjl.__author_email__
    toml["maintainer"]      = astroLuStjl.__maintainer__
    toml["maintainer_email"]= astroLuStjl.__maintainer_email__
    toml["url"]             = astroLuStjl.__url__
    toml["credits"]         = astroLuStjl.__credits__
    toml["uuid"]            = string(UUIDs.uuid4())

    println(toml)

    #add `[compat]` section
    toml["compat"] = Dict(
        "julia" => string(VERSION)              #juia version
    )

    # Write back to `Project.toml`
    open(project_toml, "w") do io
        TOML.print(io, toml)
    end

end

"""
    - function to create README.md from a template file

"""
function compile_readme()
end

"""
    - function to copy the current state of the module to the [legacy](./legacy/) directory

"""
function current2legacy(
    modulename::String, version::String="0.0.0";
    write::Bool=True, overwrite::Bool=False    
)
end

#%%control

#%%main
build_project_toml(
    "./Project.toml"
)