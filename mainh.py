import os
from agno.agent import Agent
import traceback
from agno.models.openai import OpenAIChat
from agno.models.huggingface import HuggingFace

from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.agent import RunResponse
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st
import uuid

st.set_page_config(page_title =" Article to podcast agent")
st.title("Beginner Friendly End to End Audio Podcast creator")

st.sidebar.header("API Keys")
huggingface_api_key = st.sidebar.text_input("Hugging Face API key", type='password')
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API key",type = 'password')
firecrawl_api_key = st.sidebar.text_input("Firecrawl API key",type = 'password')

keys_provided = all([huggingface_api_key,elevenlabs_api_key,firecrawl_api_key])

url = st.text_input("Enter the URL of the site","")

generate_button = st.button("Generate podcast",disabled =not keys_provided )

if not keys_provided:
    st.warning("Please enter all the keys")

if generate_button:
    if url.strip() == "":
        st.warning("Please enter a blog/post/article url")
    else:
        os.environ["HF_TOKEN"] = huggingface_api_key
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key 

        with st.spinner("Processing....,Generating podcast"):
            try:
                blog_to_podcast_agent = Agent(
                    name = "blog to podcast Agent",
                    agent_id ="blog_to_podcast_agent",
                    model = HuggingFace(id="HuggingFaceH4/zephyr-7b-beta") ,
                    tools = [
                        ElevenLabsTools(
                            #voice_id="EXAVITQu4vr4xnSDxMaL",
                            #model="eleven_multilingual_v1",
                            target_directory="audio_files"


                        ),
                        FirecrawlTools()
                    ],
                    description="This agent converts a blog post to a podcast",
                    instructions=[
                    "You are a blog to podcast agent."
                    "You will be provided with a blog post URL."
                    "Your task is to read the blog post and convert it into a podcast."

                    "You will use the Firecrawl API to read the blog post."
                    "You will use the ElevenLabs API to generate the podcast audio."

                    "You will use the Hugging Face API to generate the podcast script."
                    "You will create a concise summary of the blog post not more than 2000 characters long"
                    "You will then generate a podcast audio from the summary using ElevenLabs API.",

                    ],
                    markdown = True,
                    debug_mode=True
                )
            
                podcast :RunResponse = blog_to_podcast_agent.run(
                    f"Convert the blog post at {url} to a podcast"
                )
            
                save_dir = "audio_files"
                os.makedirs(save_dir, exist_ok=True)
            
                if podcast.audio and len(podcast.audio) > 0:
                    filename = f"{save_dir}/podcast_{uuid.uuid4()}.wav"
                    write_audio_to_file(
                        audio =podcast.audio[0].base64_audio,
                        filename = filename,
                    )

                    st.success("Podcast generated successfully!")
                    audio_bytes = open(filename, "rb").read()
                    st.audio(audio_bytes, format="audio/wav")

                    st.write("Podcast script:", podcast.text)

                    st.download_button(
                        label="Download Podcast",
                        data=audio_bytes,
                        file_name="generated_podcast.wav",
                        mime="audio/wav"
                    )

                else:
                    st.error("Failed to generate podcast audio. Please check the logs for more details.")
                    logger.error("No audio generated in the response.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.text(traceback.format_exc())
                logger.error(f"Error in generating podcast: {str(e)}")


#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