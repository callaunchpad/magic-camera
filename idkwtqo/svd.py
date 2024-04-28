import replicate


input = {
        "input_image": "https://i.ibb.co/yRrhfZ5/Screenshot-2024-04-11-at-8-46-12-PM.png"
        }
output = replicate.run(
    "sunfjun/stable-video-diffusion:d68b6e09eedbac7a49e3d8644999d93579c386a083768235cabca88796d70d82",
    input=input
)
print(output)
#=> "https://replicate.delivery/pbxt/avMWRw9yk5ImFNeCnTKsXBNY...