package io.kestra.plugin.openai;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.List;
import java.util.UUID;

import org.apache.commons.io.FileUtils;

import com.openai.client.OpenAIClient;
import com.openai.models.images.ImageGenerateParams;
import com.openai.models.images.ImageModel;
import com.openai.models.images.ImagesResponse;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import static io.kestra.core.utils.Rethrow.throwConsumer;
import io.kestra.core.models.annotations.PluginProperty;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Generate images with OpenAI",
    description = "Creates one or more images from a prompt using OpenAI Images. Default size is 1024x1024; optionally download Base64 content into Kestra internal storage or return remote URLs. See the [Images API docs](https://platform.openai.com/docs/api-reference/images/create)."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            code = """
                id: openai
                namespace: company.team

                tasks:
                  - id: create_image
                    type: io.kestra.plugin.openai.CreateImage
                    prompt: A funny cat in a black suit
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    download: true
                    n: 5
                """
        )
    }
)
public class CreateImage extends AbstractTask implements RunnableTask<CreateImage.Output> {
    @Schema(
        title = "Image prompt",
        description = "Required text sent to the Images API."
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> prompt;

    @Schema(
        title = "Number of images",
        description = "Between 1 and 10; OpenAI defaults to 1 when unset."
    )
    @PluginProperty(group = "advanced")
    private Integer n;

    @Schema(
        title = "Image size",
        description = "Defaults to 1024x1024; choose 256x256, 512x512, or 1024x1024."
    )
    @Builder.Default
    @NotNull
    @PluginProperty(group = "advanced")
    private Property<SIZE> size = Property.ofValue(SIZE.LARGE);

    @Schema(
        title = "OpenAI image model",
        description = "Model used to generate images. Defaults to `gpt-image-1`. Any image model accepted by the OpenAI API can be set, e.g. `gpt-image-1-mini`."
    )
    @Builder.Default
    @NotNull
    @PluginProperty(group = "advanced")
    private Property<String> model = Property.ofValue("gpt-image-1");

    @Schema(
        title = "Image quality",
        description = "One of `auto`, `low`, `medium`, `high`."
    )
    @PluginProperty(group = "advanced")
    private Property<QUALITY> quality;

    @Schema(
        title = "Download generated images",
        description = "Default false. When true, saves Base64 output to Kestra internal storage; when false, returns image URLs."
    )
    @Builder.Default
    @NotNull
    @PluginProperty(group = "advanced")
    private Property<Boolean> download = Property.ofValue(Boolean.FALSE);

    @Override
    public CreateImage.Output run(RunContext runContext) throws Exception {
        OpenAIClient client = this.openAIClient(runContext);

        String user = runContext.render(this.user == null ? null : runContext.render(this.user).as(String.class).orElseThrow());
        String prompt = runContext.render(this.prompt).as(String.class).orElseThrow();
        String model = runContext.render(this.model).as(String.class).orElseThrow();

        ImageGenerateParams.Builder paramsBuilder = ImageGenerateParams.builder()
            .model(ImageModel.of(model))
            .prompt(prompt)
            .size(runContext.render(this.size).as(SIZE.class).orElseThrow().getSize())
            .n(this.n)
            .user(user);

        if (this.quality != null) {
            runContext.render(this.quality).as(QUALITY.class)
                .ifPresent(q -> paramsBuilder.quality(q.getQuality()));
        }

        ImagesResponse imageResult = client.images().generate(paramsBuilder.build());

        List<URI> files = new ArrayList<>();
        imageResult.data().stream()
            .flatMap(Collection::stream)
            .forEach(throwConsumer(image ->
            {
                if (image.b64Json().isPresent()) {
                    files.add(runContext.storage().putFile(this.downloadB64Json(image.b64Json().get())));
                } else {
                    image.url().ifPresent(url -> files.add(URI.create(url)));
                }
            }));

        return Output.builder().images(files).build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {

        @Schema(
            title = "Generated images",
            description = "List of internal storage URIs or remote URLs depending on `download`."
        )
        private List<URI> images;
    }

    private File downloadB64Json(String encodedImage) throws IOException {
        File image = File.createTempFile("openai-" + UUID.randomUUID(), ".png");
        byte[] decodedBytes = Base64.getDecoder().decode(encodedImage);
        FileUtils.writeByteArrayToFile(image, decodedBytes);

        return image;
    }

    protected enum SIZE {
        SMALL("256x256", ImageGenerateParams.Size._256X256),
        MEDIUM("512x512", ImageGenerateParams.Size._512X512),
        LARGE("1024x1024", ImageGenerateParams.Size._1024X1024);

        private final String value;
        private final ImageGenerateParams.Size size;

        SIZE(String value, ImageGenerateParams.Size size) {
            this.value = value;
            this.size = size;
        }

        public ImageGenerateParams.Size getSize() {
            return size;
        }

    }

    protected enum QUALITY {
        AUTO(ImageGenerateParams.Quality.AUTO),
        LOW(ImageGenerateParams.Quality.LOW),
        MEDIUM(ImageGenerateParams.Quality.MEDIUM),
        HIGH(ImageGenerateParams.Quality.HIGH);

        private final ImageGenerateParams.Quality quality;

        QUALITY(ImageGenerateParams.Quality quality) {
            this.quality = quality;
        }

        public ImageGenerateParams.Quality getQuality() {
            return quality;
        }
    }

}
