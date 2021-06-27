module DatabaseWrapper

using PairsDB, WeyandDB

export get_data, get_database, get_type, get_param, get_uniform_param, get_ids,
    get_id, id_type

get_database(name::String, cond::Function=x->true) = PairsDB.get_database(name, cond)
get_database(name::Symbol, cond::Function=x->true) = WeyandDB.get_database("", cond)

#=
function get_database(name::String, cond::Function=x->true)
    if lowercase(name) == "weyand"
        return WeyandDB.DB(name, condition)
    else
        return PairsDB.DB(name, condition)
    end
end
=#

id_type(db::PairsDB.DB) = Int16
id_type(db::WeyandDB.DB) = Int

function PairsDB.get_data(db::WeyandDB.DB, k::Integer=0; id::Integer=-1)
    ret, lgn = WeyandDB.get_data(db, k, id=id)
    return ret, lgn, Float64[], Float64[]
end

get_id(x::Pair{Int16, Vector{String}}) = first(x)
get_id(x::Pair{Int, String}) = first(x)

get_ids(db::WeyandDB.DB) = WeyandDB.get_ids(db)
get_ids(db::PairsDB.DB) = PairsDB.get_ids(db)

end
