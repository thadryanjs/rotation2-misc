module People

export Person, say_name

    mutable struct Person

        name::String
        age::Int

    end

    function say_name(person::Person)

        println("Hello, I'm ", person.name)

    end

end