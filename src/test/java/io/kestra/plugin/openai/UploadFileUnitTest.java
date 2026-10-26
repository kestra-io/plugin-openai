package io.kestra.plugin.openai;

import org.junit.jupiter.api.Test;

import com.openai.models.files.FilePurpose;

import static org.assertj.core.api.Assertions.assertThat;

class UploadFileUnitTest {

    @Test
    void shouldMapFilePurposeCorrectly() {
        assertThat(UploadFile.getFilePurpose("assistants")).isEqualTo(FilePurpose.ASSISTANTS);
        assertThat(UploadFile.getFilePurpose("fine-tune")).isEqualTo(FilePurpose.FINE_TUNE);
        assertThat(UploadFile.getFilePurpose("vision")).isEqualTo(FilePurpose.VISION);
        assertThat(UploadFile.getFilePurpose("user_data")).isEqualTo(FilePurpose.USER_DATA);
        assertThat(UploadFile.getFilePurpose("evals")).isEqualTo(FilePurpose.EVALS);
        assertThat(UploadFile.getFilePurpose("batch")).isEqualTo(FilePurpose.BATCH);
        assertThat(UploadFile.getFilePurpose("unknown_purpose")).isEqualTo(FilePurpose.ASSISTANTS);
    }
}
